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
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._mock_text_corpus = [
            "Hello Aura, can you tell me how the live voice pipeline works?",
            "What is the latency between speech recognition and text to speech?",
            "Can you explain your Silero VAD and barge in handling?",
            "What is the weather like right now?",
            "Tell me an interesting technology joke."
        ]
        self._mock_index = 0

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

        if self.is_mock:
            await self._audio_queue.put(pcm_bytes)
            return

        if self.ws:
            try:
                await self.ws.send(pcm_bytes)
            except Exception as e:
                logger.error(f"Error streaming audio to Deepgram: {e}")

    async def receive_transcripts(self) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Receives transcription events from Deepgram.
        Yields:
            {"type": "partial" | "final", "text": str, "is_final": bool, "speech_final": bool}
        """
        if self.is_mock:
            async for event in self._mock_receive_transcripts():
                yield event
            return

        try:
            while self._connected and self.ws:
                msg = await self.ws.recv()
                if isinstance(msg, str):
                    data = json.loads(msg)
                    # Deepgram response schema
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

    async def _mock_receive_transcripts(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Simulates real-time partial transcript stream triggered by incoming audio packets."""
        target_sentence = self._mock_text_corpus[self._mock_index % len(self._mock_text_corpus)]
        self._mock_index += 1
        words = target_sentence.split()

        audio_frames_received = 0
        while self._connected:
            try:
                # Wait for audio to arrive from microphone
                await asyncio.wait_for(self._audio_queue.get(), timeout=1.0)
                audio_frames_received += 1

                # Every ~10 audio frames (~320ms), emit progressive partial words
                if audio_frames_received % 8 == 0:
                    word_count = min(len(words), (audio_frames_received // 8))
                    partial_text = " ".join(words[:word_count])
                    is_done = word_count >= len(words)
                    
                    yield {
                        "type": "final" if is_done else "partial",
                        "text": partial_text,
                        "is_final": is_done,
                        "speech_final": is_done
                    }

                    if is_done:
                        # Reset for next turn
                        audio_frames_received = 0
                        target_sentence = self._mock_text_corpus[self._mock_index % len(self._mock_text_corpus)]
                        self._mock_index += 1
                        words = target_sentence.split()
                        # Drain remaining queue
                        while not self._audio_queue.empty():
                            self._audio_queue.get_nowait()
                        await asyncio.sleep(0.5)

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    async def close(self):
        """Closes the ASR stream."""
        self._connected = False
        if self.ws:
            try:
                # Send Deepgram close stream message
                await self.ws.send(json.dumps({"type": "CloseStream"}))
                await self.ws.close()
            except Exception:
                pass
            self.ws = None
