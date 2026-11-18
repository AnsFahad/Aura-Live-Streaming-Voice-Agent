import asyncio
import logging
import uuid
from typing import Dict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from app.config import settings
from app.session import SessionContext

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("aura.main")

app = FastAPI(
    title="Aura: Live Streaming Voice Agent",
    description="Low-latency conversational voice agent using streaming ASR-LLM-TTS architecture with Silero VAD and barge-in handling.",
    version="1.0.0"
)

# Active session registry for concurrency management (supports 100+ concurrent sessions)
class ConnectionManager:
    def __init__(self, max_connections: int = 100):
        self.active_sessions: Dict[str, SessionContext] = {}
        self.max_connections = max_connections
        self.total_connections_served = 0

    async def connect(self, session_id: str, websocket: WebSocket) -> SessionContext:
        await websocket.accept()
        if len(self.active_sessions) >= self.max_connections:
            await websocket.close(code=1008, reason="Max concurrent connections reached")
            raise RuntimeError("Connection limit reached")

        session = SessionContext(session_id, websocket)
        self.active_sessions[session_id] = session
        self.total_connections_served += 1
        logger.info(f"Client connected: {session_id} (Active sessions: {len(self.active_sessions)})")
        return session

    async def disconnect(self, session_id: str):
        session = self.active_sessions.pop(session_id, None)
        if session:
            await session.cleanup()
        logger.info(f"Client disconnected: {session_id} (Active sessions: {len(self.active_sessions)})")

manager = ConnectionManager(max_connections=settings.MAX_CONCURRENT_SESSIONS)

# WebSocket endpoint for real-time voice streaming
@app.websocket("/ws/chat")
async def websocket_chat_endpoint(websocket: WebSocket):
    session_id = str(uuid.uuid4())
    session: SessionContext = None
    try:
        session = await manager.connect(session_id, websocket)
        await session.start()

        # Send initial handshake and mode status
        await session._send_control({
            "type": "connected",
            "session_id": session_id,
            "sample_rate": settings.SAMPLE_RATE,
            "modes": {
                "asr_mode": "mock" if settings.is_mock_asr else "deepgram",
                "llm_mode": "mock" if settings.is_mock_llm else "groq/llama3",
                "tts_mode": "mock" if settings.is_mock_tts else "cartesia",
                "vad_mode": "silero_onnx"
            }
        })

        # Main message processing loop
        while True:
            message = await websocket.receive()
            if "bytes" in message and message["bytes"]:
                # Incoming 16kHz PCM audio bytes
                await session.handle_audio_frame(message["bytes"])
            elif "text" in message and message["text"]:
                # Text or control commands (e.g. manual text prompt or ping)
                try:
                    import json
                    data = json.loads(message["text"])
                    if data.get("type") == "text_prompt":
                        user_text = data.get("text", "").strip()
                        if user_text:
                            session._schedule_pipeline(user_text)
                    elif data.get("type") == "client_transcript":
                        user_text = data.get("text", "").strip()
                        is_final = data.get("is_final", False)
                        if is_final and user_text:
                            # User provided a query: if assistant was speaking, trigger barge-in & answer new question!
                            if session.assistant_is_speaking:
                                await session._trigger_barge_in()
                            session.current_user_transcript = ""
                            session._schedule_pipeline(user_text)
                        else:
                            await session.asr.inject_client_transcript(user_text, is_final=False)
                    elif data.get("type") == "ping":
                        await session._send_control({"type": "pong"})
                except Exception as e:
                    logger.error(f"Error parsing incoming JSON: {e}")

    except (WebSocketDisconnect, RuntimeError):
        logger.info(f"WebSocket session closed cleanly: {session_id}")
    except Exception as e:
        logger.error(f"Unexpected error in session {session_id}: {e}")
    finally:
        if session:
            await manager.disconnect(session_id)

# API Health & Monitoring Endpoints
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "active_sessions": len(manager.active_sessions),
        "total_connections_served": manager.total_connections_served,
        "max_capacity": settings.MAX_CONCURRENT_SESSIONS
    }

@app.get("/api/config")
async def get_system_config():
    return {
        "sample_rate": settings.SAMPLE_RATE,
        "vad_model": "Silero VAD v5 ONNX",
        "asr_provider": "Deepgram (Live)" if not settings.is_mock_asr else "Deepgram (Simulation)",
        "llm_provider": f"Llama 3 ({settings.LLM_MODEL})" if not settings.is_mock_llm else "Llama 3 (Simulation)",
        "tts_provider": f"Cartesia AI ({settings.CARTESIA_MODEL})" if not settings.is_mock_tts else "Cartesia AI (Simulation)",
        "sentence_boundary_buffering": True,
        "barge_in_enabled": True
    }

# Mount static files and serve frontend
static_dir = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/")
async def serve_index():
    return FileResponse(static_dir / "index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
