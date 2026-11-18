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
        Generates realistic conversational responses with natural token streaming latency.
        """
        last_user_msg = "hello"
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user_msg = m.get("content", "").lower().strip()
                break

        # Context-aware responses without echoing the user's phrase
        if any(w in last_user_msg for w in ["hello", "hi", "hey"]):
            response_text = "Hello there! I am Aura, your live streaming voice assistant. How are you doing today?"
        elif any(w in last_user_msg for w in ["who are you", "what is aura", "what are you", "your name"]):
            response_text = "I am Aura, a conversational voice agent engineered for ultra low latency using WebSockets, Silero VAD, and sentence boundary buffering."
        elif any(w in last_user_msg for w in ["how does it work", "architecture", "latency", "pipeline"]):
            response_text = "My architecture overlaps speech recognition, token generation, and audio synthesis in parallel to deliver responses in under 500 milliseconds."
        elif any(w in last_user_msg for w in ["barge in", "interrupt", "vad"]):
            response_text = "Silero VAD monitors incoming audio frames. If you speak while I am talking, I stop immediately so you have full control."
        elif any(w in last_user_msg for w in ["weather", "temperature", "rain"]):
            response_text = "The forecast looks great right now. Mild temperatures and clear skies!"
        elif any(w in last_user_msg for w in ["joke", "funny"]):
            response_text = "Why did the developer go broke? Because they used up all their cache!"
        elif any(w in last_user_msg for w in ["thank", "thanks"]):
            response_text = "You are very welcome! Let me know if there is anything else you need."
        elif any(w in last_user_msg for w in ["bye", "goodbye"]):
            response_text = "Goodbye! It was great talking with you. Have a wonderful day!"
        else:
            response_text = "That sounds interesting. Tell me more, or ask me any question about the voice pipeline."

        words = response_text.split(" ")
        for i, word in enumerate(words):
            token = word if i == len(words) - 1 else word + " "
            yield token
            # Paced token generation (~25ms per token)
            await asyncio.sleep(0.025)
