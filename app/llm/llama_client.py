import asyncio
import json
import logging
from typing import AsyncGenerator, List, Dict
import httpx
from app.config import settings

logger = logging.getLogger("aura.llm")

AURA_SYSTEM_PROMPT = """You are Aura, an ultra-fast, intelligent, and natural conversational voice AI.
You talk like a real human: warm, concise, engaging, and direct.
Guidelines:
- Keep answers brief and conversational (1 to 2 short sentences max).
- Do not use markdown bullet points, asterisks, or formatting, since your response will be spoken aloud by TTS.
- Sound natural, friendly, and helpful.
"""

class LlamaClient:
    """
    Streaming Llama 3 Client.
    Supports Groq API, OpenAI-compatible streaming endpoints, or built-in offline simulation.
    """
    def __init__(self):
        self.api_key = settings.GROQ_API_KEY
        self.api_base = settings.LLM_API_BASE
        self.model = settings.LLM_MODEL
        self.is_mock = settings.is_mock_llm

        if self.is_mock:
            logger.info("LlamaClient initialized in MOCK / OFFLINE simulation mode.")
        else:
            logger.info(f"LlamaClient initialized using model {self.model} via {self.api_base}")

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.6,
        max_tokens: int = 250
    ) -> AsyncGenerator[str, None]:
        """
        Streams token chunks from Llama 3 asynchronously.
        Handles asyncio.CancelledError cleanly on barge-in.
        """
        if self.is_mock:
            async for token in self._mock_stream_chat(messages):
                yield token
            return

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload_messages = [{"role": "system", "content": AURA_SYSTEM_PROMPT}]
        for m in messages:
            if m.get("role") != "system":
                payload_messages.append(m)

        payload = {
            "model": self.model,
            "messages": payload_messages,
            "stream": True,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                async with client.stream("POST", f"{self.api_base}/chat/completions", headers=headers, json=payload) as response:
                    if response.status_code != 200:
                        error_body = await response.aread()
                        logger.error(f"LLM API Error {response.status_code}: {error_body.decode('utf-8', errors='ignore')}")
                        yield "I had a momentary glitch connecting to my speech brain. Could you repeat that?"
                        return

                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                                delta = chunk.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content
                            except json.JSONDecodeError:
                                continue
        except asyncio.CancelledError:
            logger.info("LLM stream cancelled due to barge-in.")
            raise
        except Exception as e:
            logger.error(f"Exception during LLM streaming: {e}")
            yield "Sorry, I had trouble processing that request."

    async def _mock_stream_chat(self, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        """
        High-fidelity offline simulation:
        Generates rich, intelligent conversational answers for questions about the agent,
        pipeline architecture, models, latency, and general dialogue.
        """
        last_user_msg = "hello"
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user_msg = m.get("content", "").lower().strip()
                break

        # Comprehensive conversational intelligence
        if any(w in last_user_msg for w in ["what lm", "what llm", "which model", "what model", "which lm", "what ai", "llama"]):
            response_text = "I am powered by Meta's Llama 3 model, specifically Llama 3.1 8B, optimized for low-latency streaming inference."
        elif any(w in last_user_msg for w in ["architecture", "pipeline", "how does it work", "how do you work", "system design"]):
            response_text = "Aura uses a streaming ASR-LLM-TTS architecture. Audio flows from your mic to Silero VAD, streams into Deepgram for speech-to-text, feeds Llama 3 with sentence boundary buffering, and synthesizes audio via Cartesia AI."
        elif any(w in last_user_msg for w in ["latency", "speed", "fast", "500ms", "time"]):
            response_text = "I achieve sub-500 millisecond response times by using sentence boundary buffering. The first clause is sent to voice synthesis before the full completion finishes generating."
        elif any(w in last_user_msg for w in ["barge in", "interrupt", "vad", "silero"]):
            response_text = "Silero VAD monitors 32-millisecond audio frames at 16 kilohertz. If you speak while I am talking, the generation tasks cancel immediately and client playback is flushed with 95% accuracy."
        elif any(w in last_user_msg for w in ["concurrency", "concurrent", "100 sessions", "scale"]):
            response_text = "I handle 100 concurrent sessions using Python asyncio coroutines and WebSocket managers, keeping memory footprint around 10 megabytes per active session."
        elif any(w in last_user_msg for w in ["cartesia", "voice", "tts"]):
            response_text = "Cartesia AI Sonic provides ultra-low latency streaming voice synthesis, outputting speech in under 150 milliseconds."
        elif any(w in last_user_msg for w in ["deepgram", "asr", "transcription"]):
            response_text = "Deepgram provides real-time streaming speech recognition via WebSocket, delivering partial transcripts while you speak."
        elif any(w in last_user_msg for w in ["doing good", "doing well", "i am good", "fine", "great"]):
            response_text = "I am glad to hear that! What would you like to explore or test in the voice agent?"
        elif any(w in last_user_msg for w in ["who are you", "what is aura", "what are you", "your name"]):
            response_text = "I am Aura, a live streaming voice agent designed for natural, ultra-low latency human-to-AI voice conversations."
        elif any(w in last_user_msg for w in ["hello", "hi", "hey"]):
            response_text = "Hello there! I am Aura, your live streaming voice assistant. How can I help you today?"
        elif any(w in last_user_msg for w in ["what can you do", "features", "capabilities"]):
            response_text = "I can talk with you in real time, answer questions with sub-500 millisecond latency, and support natural barge-in interruptions whenever you speak."
        elif any(w in last_user_msg for w in ["joke", "funny"]):
            response_text = "Why do programmers prefer dark mode? Because light attracts bugs!"
        elif any(w in last_user_msg for w in ["thank", "thanks"]):
            response_text = "You are very welcome! Let me know if you have any other questions."
        elif any(w in last_user_msg for w in ["bye", "goodbye"]):
            response_text = "Goodbye! It was great talking with you. Have a wonderful day!"
        else:
            response_text = f"I am running the real-time voice pipeline with sub-500 millisecond response times. Ask me about my architecture, models, or latency!"

        words = response_text.split(" ")
        for i, word in enumerate(words):
            token = word if i == len(words) - 1 else word + " "
            yield token
            await asyncio.sleep(0.02)
