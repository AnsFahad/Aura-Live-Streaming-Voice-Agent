import os
from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load .env if present
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

class Settings(BaseSettings):
    # App Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    MAX_CONCURRENT_SESSIONS: int = 100

    # Audio Configuration
    SAMPLE_RATE: int = 16000
    FRAME_SIZE: int = 512              # 512 samples = 32ms at 16kHz
    CHANNELS: int = 1
    BYTES_PER_SAMPLE: int = 2          # 16-bit linear PCM = 2 bytes

    # Silero VAD Configuration
    VAD_MODEL_PATH: str = str(Path(__file__).resolve().parent / "vad" / "models" / "silero_vad.onnx")
    VAD_THRESHOLD: float = 0.5
    VAD_SPEECH_PAD_FRAMES: int = 2
    VAD_MIN_SPEECH_DURATION_MS: int = 100
    VAD_SILENCE_TIMEOUT_MS: int = 400  # Snappy 400ms silence timeout for instant turn completion
    VAD_BARGE_IN_CONSECUTIVE_FRAMES: int = 2  # ~64ms consecutive speech to trigger automatic voice barge-in

    # Deepgram ASR Configuration
    DEEPGRAM_API_KEY: str = os.getenv("DEEPGRAM_API_KEY", "")
    DEEPGRAM_MODEL: str = "nova-2"
    DEEPGRAM_WS_URL: str = "wss://api.deepgram.com/v1/listen"

    # LLM (Groq / OpenAI-compatible endpoint)
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    LLM_API_BASE: str = os.getenv("LLM_API_BASE", "https://api.groq.com/openai/v1")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen/qwen3.8-27b")
    LLM_TEMPERATURE: float = 0.6
    LLM_MAX_TOKENS: int = 500

    # Cartesia AI TTS Configuration
    CARTESIA_API_KEY: str = os.getenv("CARTESIA_API_KEY", "")
    CARTESIA_VOICE_ID: str = os.getenv("CARTESIA_VOICE_ID", "a0e99841-438c-4a64-b679-ae501e7d6091")  # Barbershop Man / British/American natural
    CARTESIA_MODEL: str = "sonic-english"
    CARTESIA_WS_URL: str = "wss://api.cartesia.ai/tts/websocket"

    # Simulation / Mock Mode
    FORCE_MOCK: bool = os.getenv("FORCE_MOCK", "false").lower() in ("true", "1", "yes")

    @property
    def is_mock_asr(self) -> bool:
        return self.FORCE_MOCK or not self.DEEPGRAM_API_KEY or self.DEEPGRAM_API_KEY.startswith("your_")

    @property
    def is_mock_llm(self) -> bool:
        return self.FORCE_MOCK or not self.GROQ_API_KEY or self.GROQ_API_KEY.startswith("your_")

    @property
    def is_mock_tts(self) -> bool:
        return self.FORCE_MOCK or not self.CARTESIA_API_KEY or self.CARTESIA_API_KEY.startswith("your_")

settings = Settings()
