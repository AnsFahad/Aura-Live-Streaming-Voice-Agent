import asyncio
import json
import logging
import math
import struct
from typing import AsyncGenerator, Optional
import websockets
from app.config import settings

logger = logging.getLogger("aura.tts")

class CartesiaStreamingTTS:
    """
    Streaming Text-to-Speech client for Cartesia AI Sonic API via WebSocket.
    Accepts text clauses/sentences and yields raw 16kHz 16-bit PCM audio chunks.
    """
    def __init__(self, context_id: str):
        self.context_id = context_id
        self.api_key = settings.CARTESIA_API_KEY
        self.voice_id = settings.CARTESIA_VOICE_ID
        self.model_id = settings.CARTESIA_MODEL
        self.is_mock = settings.is_mock_tts
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self._connected = False

    async def connect(self):
        """Connects to Cartesia TTS WebSocket."""
        if self.is_mock:
            self._connected = True
            logger.info("CartesiaStreamingTTS connected in MOCK / OFFLINE mode.")
            return

        url = f"{settings.CARTESIA_WS_URL}?api_key={self.api_key}&cartesia_version=2024-06-10"
        try:
            self.ws = await websockets.connect(url)
            self._connected = True
            logger.info("Connected to Cartesia AI Sonic WebSocket.")
        except Exception as e:
            logger.error(f"Failed to connect to Cartesia: {e}. Falling back to mock TTS.")
            self.is_mock = True
            self._connected = True

    async def stream_speech(self, text_chunk: str, is_last_chunk: bool = False) -> AsyncGenerator[bytes, None]:
        """
        Sends a sentence/clause to Cartesia and streams back raw audio PCM chunks.
        Handles barge-in cancellation via asyncio.CancelledError.
        """
        if not text_chunk.strip():
            return

        if self.is_mock:
            async for audio_chunk in self._mock_stream_speech(text_chunk):
                yield audio_chunk
            return

        payload = {
            "context_id": self.context_id,
            "model_id": self.model_id,
            "transcript": text_chunk,
            "voice": {
                "mode": "id",
                "id": self.voice_id
            },
            "output_format": {
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": settings.SAMPLE_RATE
            },
            "continue": not is_last_chunk
        }

        try:
            if not self.ws:
                await self.connect()

            await self.ws.send(json.dumps(payload))

            while True:
                msg = await self.ws.recv()
                if isinstance(msg, bytes):
                    # Binary PCM chunk
                    yield msg
                elif isinstance(msg, str):
                    data = json.loads(msg)
                    if data.get("type") == "chunk":
                        # Some versions send base64 audio in JSON
                        import base64
                        b64_audio = data.get("data", "")
                        if b64_audio:
                            yield base64.b64decode(b64_audio)
                    elif data.get("done", False) or data.get("type") == "done":
                        break
        except asyncio.CancelledError:
            logger.info("Cartesia TTS synthesis cancelled due to barge-in.")
            # Clear Cartesia server context if connected
            if self.ws and not self.is_mock:
                try:
                    await self.ws.send(json.dumps({"context_id": self.context_id, "cancel": True}))
                except Exception:
                    pass
            raise
        except Exception as e:
            logger.error(f"Error in Cartesia TTS stream: {e}")

    async def _mock_stream_speech(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Synthesizes smooth, natural-sounding audio frames (formant modulated waveform)
        for offline testing. Paced at actual audio playback rate.
        """
        words = text.split()
        sample_rate = settings.SAMPLE_RATE
        chunk_duration_s = 0.05  # 50ms audio chunk
        samples_per_chunk = int(sample_rate * chunk_duration_s)

        # Base pitch and harmonics for pleasant vocal simulation
        base_freq = 220.0  # A3 pitch
        time_counter = 0.0

        for word_idx, word in enumerate(words):
            word_duration = max(0.18, len(word) * 0.04)  # ~180-350ms per word
            chunks_for_word = int(word_duration / chunk_duration_s)
            
            # Subtle pitch inflection based on punctuation
            inflection = 1.15 if word.endswith("?") else (0.9 if word.endswith(".") else 1.0)
            current_pitch = base_freq * (1.0 + (word_idx % 4) * 0.05) * inflection

            for c in range(chunks_for_word):
                samples = []
                for s in range(samples_per_chunk):
                    t = time_counter + (s / sample_rate)
                    # Envelope modulation to avoid clicks
                    envelope = min(1.0, 4.0 * (s / samples_per_chunk)) * min(1.0, 4.0 * ((samples_per_chunk - s) / samples_per_chunk))
                    # Fundamental + 2nd harmonic (warm human vocal formant approximation)
                    v = 0.4 * math.sin(2 * math.pi * current_pitch * t) + \
                        0.2 * math.sin(2 * math.pi * (current_pitch * 2) * t) + \
                        0.1 * math.sin(2 * math.pi * (current_pitch * 3) * t)
                    v = max(-1.0, min(1.0, v * envelope))
                    samples.append(int(v * 28000))

                time_counter += chunk_duration_s
                pcm_data = struct.pack(f"<{len(samples)}h", *samples)
                yield pcm_data
                # Deliver at true real-time audio playback rate
                await asyncio.sleep(chunk_duration_s * 0.95)

    async def close(self):
        """Closes TTS connection."""
        self._connected = False
        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass
            self.ws = None
