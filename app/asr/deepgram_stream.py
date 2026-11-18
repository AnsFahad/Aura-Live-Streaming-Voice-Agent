import asyncio
import json
import logging
from typing import AsyncGenerator, Dict, Any, Optional
import websockets
from app.config import settings

logger = logging.getLogger("aura.asr")

class DeepgramStreamingASR:
    """
    Streaming Automatic Speech Recognition client using Deepgram WebSocket API.
    Streams 16kHz linear PCM audio frames and yields real-time partial and final transcripts.
    """
    def __init__(self):
        self.api_key = settings.DEEPGRAM_API_KEY
        self.is_mock = settings.is_mock_asr
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self._connected = False
        self._speech_event_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()

    async def connect(self):
        """Establishes streaming WebSocket to Deepgram."""
        if self.is_mock:
            self._connected = True
            logger.info("DeepgramStreamingASR connected in MOCK / OFFLINE mode.")
            return

        url = (
            f"{settings.DEEPGRAM_WS_URL}?"
            f"model={settings.DEEPGRAM_MODEL}&"
            f"encoding=linear16&"
            f"sample_rate={settings.SAMPLE_RATE}&"
            f"channels={settings.CHANNELS}&"
            f"interim_results=true&"
            f"smart_format=true&"
            f"endpointing=300&"
            f"vad_events=true"
        )
        headers = {"Authorization": f"Token {self.api_key}"}
        
        try:
            self.ws = await websockets.connect(url, additional_headers=headers)
            self._connected = True
            logger.info("Connected to Deepgram Live Streaming ASR WebSocket.")
        except Exception as e:
            logger.error(f"Failed to connect to Deepgram: {e}. Falling back to mock ASR.")
            self.is_mock = True
            self._connected = True

    async def send_audio(self, pcm_bytes: bytes):
        """Pushes raw PCM audio frames to ASR."""
        if not self._connected:
            return

        if not self.is_mock and self.ws:
            try:
                await self.ws.send(pcm_bytes)
            except Exception as e:
                logger.error(f"Error streaming audio to Deepgram: {e}")

    async def inject_client_transcript(self, text: str, is_final: bool):
        """Allows client-side Web Speech API to provide transcript in simulation mode."""
        if self.is_mock:
            await self._speech_event_queue.put({
                "type": "final" if is_final else "partial",
                "text": text,
                "is_final": is_final,
                "speech_final": is_final
            })

    async def receive_transcripts(self) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Receives transcription events.
        Yields:
            {"type": "partial" | "final", "text": str, "is_final": bool, "speech_final": bool}
        """
        if self.is_mock:
            while self._connected:
                try:
                    event = await self._speech_event_queue.get()
                    yield event
                except asyncio.CancelledError:
                    break
            return

        try:
            while self._connected and self.ws:
                msg = await self.ws.recv()
                if isinstance(msg, str):
                    data = json.loads(msg)
                    channel = data.get("channel", {})
                    alternatives = channel.get("alternatives", [])
                    if alternatives:
                        transcript = alternatives[0].get("transcript", "").strip()
                        is_final = data.get("is_final", False)
                        speech_final = data.get("speech_final", False)

                        if transcript:
                            yield {
                                "type": "final" if is_final else "partial",
                                "text": transcript,
                                "is_final": is_final,
                                "speech_final": speech_final
                            }
        except asyncio.CancelledError:
            logger.info("Deepgram receiver task cancelled.")
            raise
        except Exception as e:
            logger.error(f"Deepgram receive error: {e}")

    async def close(self):
        """Closes the ASR stream."""
        self._connected = False
        if self.ws:
            try:
                await self.ws.send(json.dumps({"type": "CloseStream"}))
                await self.ws.close()
            except Exception:
                pass
            self.ws = None
