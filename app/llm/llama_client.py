import asyncio
import json
import logging
from typing import AsyncGenerator, List, Dict
import httpx
from app.config import settings

logger = logging.getLogger("aura.llm")

# System prompt optimized for low-latency conversational voice agent
AURA_SYSTEM_PROMPT = """You are Aura, an ultra-fast, intelligent, and natural conversational voice AI.
You talk like a real human: warm, concise, engaging, and direct.
Guidelines:
- Keep answers brief and conversational (1 to 3 short sentences max).
- Do not use markdown bullet points, asterisks, or formatting, since your response will be spoken aloud by TTS.
- Sound natural and confident.
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

        # Ensure system prompt is present
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
        Generates realistic contextual conversational replies with realistic token streaming latency.
        """
        last_user_msg = "hello"
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user_msg = m.get("content", "").lower()
                break

        # Context-aware simulated responses
        if any(w in last_user_msg for w in ["who are you", "what is aura", "what are you"]):
            response_text = "I am Aura, a low-latency live streaming voice agent. I can talk with you in real-time, process speech with sub-500 millisecond response times, and handle interruptions naturally."
        elif any(w in last_user_msg for w in ["how does it work", "architecture", "latency", "vad", "barge in"]):
            response_text = "I run on an asynchronous streaming pipeline with Silero VAD, Deepgram speech recognition, Llama 3, and Cartesia voice synthesis. My sentence boundary buffer begins speaking before the full thought finishes generating."
        elif any(w in last_user_msg for w in ["weather", "temperature"]):
            response_text = "It feels crisp and clear right now. Perfect weather for testing out low-latency speech pipelines!"
        elif any(w in last_user_msg for w in ["joke", "funny"]):
            response_text = "Why did the neural network cross the road? To optimize the loss function on the other side!"
        else:
            response_text = f"I hear you loud and clear! You said '{last_user_msg}'. Everything in the real-time audio pipeline is streaming seamlessly with sub-500 millisecond latency."

        words = response_text.split(" ")
        for i, word in enumerate(words):
            token = word if i == len(words) - 1 else word + " "
            yield token
            # Simulate ultra-fast token streaming (~25ms per token = ~40 words/sec)
            await asyncio.sleep(0.025)
