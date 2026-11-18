import asyncio
import json
import logging
import time
from typing import List, Dict, Optional
from fastapi import WebSocket

from app.config import settings
from app.vad.silero import VoiceActivityDetector
from app.asr.deepgram_stream import DeepgramStreamingASR
from app.llm.llama_client import LlamaClient
from app.llm.sentence_buffer import SentenceBoundaryBuffer
from app.tts.cartesia_stream import CartesiaStreamingTTS

logger = logging.getLogger("aura.session")

class SessionContext:
    """
    Manages an isolated, concurrent conversational voice session.
    Orchestrates the real-time VAD -> ASR -> LLM -> SentenceBuffer -> TTS pipeline
    with asynchronous barge-in cancellation and latency benchmarking.
    """
    def __init__(self, session_id: str, client_ws: WebSocket):
        self.session_id = session_id
        self.client_ws = client_ws

        # Component instances
        self.vad = VoiceActivityDetector(
            model_path=settings.VAD_MODEL_PATH,
            threshold=settings.VAD_THRESHOLD,
            sample_rate=settings.SAMPLE_RATE,
            frame_size=settings.FRAME_SIZE,
            silence_timeout_ms=settings.VAD_SILENCE_TIMEOUT_MS,
            barge_in_consecutive_frames=settings.VAD_BARGE_IN_CONSECUTIVE_FRAMES
        )
        self.asr = DeepgramStreamingASR()
        self.llm = LlamaClient()
        self.sentence_buffer = SentenceBoundaryBuffer()
        self.tts = CartesiaStreamingTTS(context_id=session_id)

        # State tracking
        self.is_active = True
        self.assistant_is_speaking = False
        self.conversation_history: List[Dict[str, str]] = []
        self.current_user_transcript = ""

        # Background tasks
        self.asr_task: Optional[asyncio.Task] = None
        self.pipeline_task: Optional[asyncio.Task] = None

        # Latency benchmarking
        self.speech_end_timestamp: float = 0.0
        self.first_token_timestamp: float = 0.0
        self.first_audio_timestamp: float = 0.0

        # Metrics
        self.barge_in_count: int = 0
        self.total_turns: int = 0

    async def start(self):
        """Initializes ASR and TTS connections and starts background listener tasks."""
        await self.asr.connect()
        await self.tts.connect()
        self.asr_task = asyncio.create_task(self._listen_to_asr())
        logger.info(f"Session {self.session_id} started.")

    async def handle_audio_frame(self, pcm_bytes: bytes):
        """Processes incoming 16kHz PCM audio bytes from client."""
        if not self.is_active:
            return

        # 1. Silero VAD evaluation
        prob, turn_completed, should_barge_in = self.vad.process_pcm_bytes(
            pcm_bytes,
            assistant_is_speaking=self.assistant_is_speaking
        )

        # 2. Barge-in handling
        if should_barge_in and self.assistant_is_speaking:
            await self._trigger_barge_in()

        # 3. Stream audio frame to ASR
        await self.asr.send_audio(pcm_bytes)

        # 4. Notify client of VAD speech probability for visualizer
        if prob is not None:
            await self._send_control({
                "type": "vad_activity",
                "speech_prob": round(prob, 3),
                "is_speaking": self.vad.is_speech_active
            })

        # 5. Turn completion detected by VAD (fallback if ASR endpointing is delayed)
        if turn_completed and self.current_user_transcript.strip():
            logger.info(f"VAD detected end of user speech: '{self.current_user_transcript}'")
            self.speech_end_timestamp = time.perf_counter()
            final_text = self.current_user_transcript
            self.current_user_transcript = ""
            self._schedule_pipeline(final_text)

    async def _listen_to_asr(self):
        """Consumes real-time transcripts from Deepgram."""
        try:
            async for event in self.asr.receive_transcripts():
                text = event.get("text", "")
                is_final = event.get("is_final", False)
                speech_final = event.get("speech_final", False)

                if not text:
                    continue

                if is_final:
                    self.current_user_transcript = text
                    await self._send_control({
                        "type": "user_transcript",
                        "text": text,
                        "is_final": True
                    })
                else:
                    self.current_user_transcript = text
                    await self._send_control({
                        "type": "user_transcript",
                        "text": text,
                        "is_final": False
                    })

                # If speech concluded, dispatch to LLM
                if speech_final or (is_final and not self.vad.is_speech_active):
                    if self.current_user_transcript.strip():
                        self.speech_end_timestamp = time.perf_counter()
                        final_text = self.current_user_transcript
                        self.current_user_transcript = ""
                        self._schedule_pipeline(final_text)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in ASR listener for session {self.session_id}: {e}")

    def _schedule_pipeline(self, user_text: str):
        """Cancels any stale pipeline task and schedules a new generation turn."""
        if self.pipeline_task and not self.pipeline_task.done():
            self.pipeline_task.cancel()
        self.pipeline_task = asyncio.create_task(self._run_generation_pipeline(user_text))

    async def _run_generation_pipeline(self, user_text: str):
        """
        Orchestrates LLM generation -> Sentence Boundary Buffering -> Streaming TTS -> Audio Playback.
        """
        self.total_turns += 1
        self.assistant_is_speaking = True
        self.first_token_timestamp = 0.0
        self.first_audio_timestamp = 0.0

        start_time = self.speech_end_timestamp or time.perf_counter()
        logger.info(f"Starting response generation for: '{user_text}'")

        # Keep sliding window of last 4 messages to prevent prompt token accumulation & latency drift
        if len(self.conversation_history) > 4:
            self.conversation_history = self.conversation_history[-4:]

        self.conversation_history.append({"role": "user", "content": user_text})
        self.sentence_buffer.reset()

        full_assistant_reply = ""
        spoken_sentences: List[str] = []

        try:
            # 1. Stream tokens from Llama 3
            token_stream = self.llm.stream_chat(self.conversation_history)
            
            async for token in token_stream:
                if not self.first_token_timestamp:
                    self.first_token_timestamp = time.perf_counter()
                    ttft_ms = round((self.first_token_timestamp - start_time) * 1000, 1)
                    await self._send_control({
                        "type": "metric_ttft",
                        "ttft_ms": ttft_ms
                    })

                full_assistant_reply += token

                # Send partial assistant text to UI
                await self._send_control({
                    "type": "assistant_transcript",
                    "text": full_assistant_reply,
                    "is_final": False
                })

                # 2. Sentence Boundary Buffering: Extract complete sentences/clauses
                sentences = self.sentence_buffer.process_token(token)
                for sentence in sentences:
                    spoken_sentences.append(sentence)
                    await self._synthesize_and_stream_audio(sentence, is_last=False, start_time=start_time)

            # Flush remaining buffer tokens
            remaining_sentences = self.sentence_buffer.flush()
            for i, rem_sentence in enumerate(remaining_sentences):
                spoken_sentences.append(rem_sentence)
                is_last = (i == len(remaining_sentences) - 1)
                await self._synthesize_and_stream_audio(rem_sentence, is_last=is_last, start_time=start_time)

            # Finalize assistant turn in history
            self.conversation_history.append({"role": "assistant", "content": full_assistant_reply.strip()})
            await self._send_control({
                "type": "assistant_transcript",
                "text": full_assistant_reply.strip(),
                "is_final": True
            })

        except asyncio.CancelledError:
            logger.info("Pipeline cancelled mid-generation by barge-in.")
            if spoken_sentences:
                actual_spoken = " ".join(spoken_sentences)
                self.conversation_history.append({"role": "assistant", "content": actual_spoken})
            raise
        finally:
            self.assistant_is_speaking = False

    async def _synthesize_and_stream_audio(self, sentence: str, is_last: bool, start_time: float):
        """Streams a sentence chunk through Cartesia TTS or speaks via natural voice."""
        logger.debug(f"Synthesizing sentence chunk: '{sentence}'")
        
        # Check Time-To-First-Audio (TTFA)
        if not self.first_audio_timestamp:
            self.first_audio_timestamp = time.perf_counter()
            ttfa_ms = round((self.first_audio_timestamp - start_time) * 1000, 1)
            logger.info(f"⚡ Time-To-First-Audio (Response Latency): {ttfa_ms}ms")
            await self._send_control({
                "type": "metric_ttfa",
                "ttfa_ms": ttfa_ms
            })

        if settings.is_mock_tts:
            # Deliver sentence chunk to client immediately for instant voice playback
            await self._send_control({
                "type": "speak_sentence",
                "text": sentence,
                "is_last": is_last
            })
            return

        async for audio_chunk in self.tts.stream_speech(sentence, is_last_chunk=is_last):
            await self.client_ws.send_bytes(audio_chunk)

    async def _trigger_barge_in(self):
        """
        Executes instant barge-in interruption:
        1. Cancels active generation pipeline task.
        2. Sends interrupt frame to client to dump playback audio queue immediately.
        3. Updates metrics.
        """
        logger.info("🚨 BARGE-IN TRIGGERED: User interrupted assistant playback!")
        self.barge_in_count += 1
        self.assistant_is_speaking = False

        if self.pipeline_task and not self.pipeline_task.done():
            self.pipeline_task.cancel()

        # Send immediate interrupt frame to client
        await self._send_control({
            "type": "interrupt",
            "message": "Playback interrupted by user speech",
            "barge_in_count": self.barge_in_count,
            "timestamp": time.time()
        })

    async def _send_control(self, payload: dict):
        """Helper to send JSON control messages to client WebSocket."""
        try:
            await self.client_ws.send_text(json.dumps(payload))
        except Exception:
            pass

    async def cleanup(self):
        """Cleans up all session tasks and network resources."""
        self.is_active = False
        if self.pipeline_task and not self.pipeline_task.done():
            self.pipeline_task.cancel()
        if self.asr_task and not self.asr_task.done():
            self.asr_task.cancel()

        await self.asr.close()
        await self.tts.close()
        logger.info(f"Session {self.session_id} cleaned up.")
