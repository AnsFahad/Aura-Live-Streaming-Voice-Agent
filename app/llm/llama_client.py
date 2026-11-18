import asyncio
import json
import logging
import re
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
        High-fidelity conversational simulation engine.
        Answers questions about the resume project, models, science, tech, general knowledge,
        and everyday dialogue naturally.
        """
        last_user_msg = "hello"
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user_msg = m.get("content", "").lower().strip()
                break

        # 1. Models & Tech Stack
        if any(w in last_user_msg for w in ["what lm", "what llm", "which model", "what model", "which lm", "what ai", "llama"]):
            response_text = "I am powered by Meta's Llama 3 model, specifically Llama 3.1 8B, optimized for low-latency streaming inference."
        elif any(w in last_user_msg for w in ["cartesia", "voice", "tts", "sonic"]):
            response_text = "Cartesia AI Sonic provides ultra-low latency streaming voice synthesis, outputting speech chunks in under 150 milliseconds."
        elif any(w in last_user_msg for w in ["deepgram", "asr", "transcription", "speech to text", "stt"]):
            response_text = "Deepgram provides real-time streaming speech recognition via WebSocket, delivering word-level partial transcripts as you speak."
        elif any(w in last_user_msg for w in ["silero", "vad", "voice activity"]):
            response_text = "Silero VAD v5 runs locally via ONNX on 32-millisecond audio frames to detect speech onsets and power 95% accurate barge-in."

        # 2. Architecture & Resume Points
        elif any(w in last_user_msg for w in ["architecture", "pipeline", "how does it work", "how do you work", "system design"]):
            response_text = "Aura uses a streaming ASR-LLM-TTS architecture. Audio flows from your mic to Silero VAD, streams into Deepgram for speech-to-text, feeds Llama 3 with sentence boundary buffering, and synthesizes audio via Cartesia AI."
        elif any(w in last_user_msg for w in ["latency", "speed", "fast", "500ms", "time to first audio", "ttfa"]):
            response_text = "I achieve sub-500 millisecond response times by using sentence boundary buffering. The first clause is sent to voice synthesis before the full completion finishes generating."
        elif any(w in last_user_msg for w in ["barge in", "interrupt", "interruption"]):
            response_text = "Silero VAD monitors incoming audio frames in real time. If you speak while I am talking, generation tasks cancel immediately and client playback is flushed without needing to press any button."
        elif any(w in last_user_msg for w in ["concurrency", "concurrent", "100 sessions", "scale"]):
            response_text = "I handle 100 concurrent sessions using Python asyncio coroutines and WebSocket managers, keeping memory footprint around 10 megabytes per active session."

        # 3. Simple Math & Facts
        elif any(w in last_user_msg for w in ["plus", "minus", "times", "divided by", "math", "+", "-", "*", "/"]):
            # Simple arithmetic evaluator
            try:
                clean_expr = re.sub(r'[^0-9+\-*/.]', '', last_user_msg)
                if clean_expr:
                    ans = eval(clean_expr)
                    response_text = f"The answer is {ans}."
                else:
                    response_text = "I can do math! What numbers would you like me to calculate?"
            except Exception:
                response_text = "I can calculate that for you. Could you rephrase the numbers clearly?"
        elif any(w in last_user_msg for w in ["capital of france", "capital of india", "capital of usa", "capital"]):
            if "france" in last_user_msg: response_text = "The capital of France is Paris."
            elif "india" in last_user_msg: response_text = "The capital of India is New Delhi."
            elif "usa" in last_user_msg or "united states" in last_user_msg: response_text = "The capital of the United States is Washington, D.C."
            else: response_text = "Which country or state's capital are you curious about?"
        elif any(w in last_user_msg for w in ["speed of light"]):
            response_text = "The speed of light in a vacuum is approximately 299,792 kilometers per second, or about 186,000 miles per second."
        elif any(w in last_user_msg for w in ["python", "programming", "code", "coding"]):
            response_text = "Python is a powerful, expressive language ideal for real-time async pipelines, machine learning, and low-latency audio processing."
        elif any(w in last_user_msg for w in ["what is ai", "artificial intelligence"]):
            response_text = "Artificial intelligence refers to computer systems capable of performing tasks that normally require human intelligence, like speech recognition, reasoning, and voice generation."

        # 4. Greetings & Conversational
        elif any(w in last_user_msg for w in ["who are you", "what is aura", "what are you", "your name"]):
            response_text = "I am Aura, a live streaming voice agent designed for natural, ultra-low latency human-to-AI voice conversations."
        elif any(w in last_user_msg for w in ["hello", "hi", "hey"]):
            response_text = "Hello there! I am Aura, your live streaming voice assistant. How can I help you today?"
        elif any(w in last_user_msg for w in ["doing good", "doing well", "i am good", "fine", "great"]):
            response_text = "I am glad to hear that! What would you like to discuss or test next?"
        elif any(w in last_user_msg for w in ["what can you do", "features", "capabilities"]):
            response_text = "I can talk with you in real time, answer questions with sub-500 millisecond latency, and support natural voice barge-in interruptions whenever you speak."
        elif any(w in last_user_msg for w in ["joke", "funny"]):
            response_text = "Why did the developer go broke? Because they used up all their cache!"
        elif any(w in last_user_msg for w in ["thank", "thanks"]):
            response_text = "You are very welcome! Ask me anything else anytime."
        elif any(w in last_user_msg for w in ["bye", "goodbye"]):
            response_text = "Goodbye! It was great talking with you. Have a wonderful day!"
        else:
            # Context-aware fallback based on question structure
            if last_user_msg.startswith(("why", "how", "what", "where", "who", "when", "can you", "is ")):
                response_text = f"That is a great question. In a live production setup with my Llama 3 API connected, I provide full in-depth answers across science, history, coding, and general knowledge. Ask me about the pipeline architecture or voice metrics!"
            else:
                response_text = f"I hear you! I am streaming with sub-500 millisecond response times. Feel free to ask me anything about how the live pipeline works or test barge-in interruptions."

        words = response_text.split(" ")
        for i, word in enumerate(words):
            token = word if i == len(words) - 1 else word + " "
            yield token
            await asyncio.sleep(0.015)
