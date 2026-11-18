# Aura: Live Streaming Voice Agent

> **Low-latency conversational voice agent using a streaming ASR-LLM-TTS architecture.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Aura is a real-time conversational voice agent designed for ultra-low latency human-to-AI dialogue. Instead of waiting for complete LLM generation, Aura pipelines speech recognition, token generation, and voice synthesis concurrently using **Sentence Boundary Buffering (SBB)** to achieve **sub-500ms Time-To-First-Audio (TTFA)**.

---

## ✨ Key Features & Technical Specifications

| Feature | Technical Implementation | Code References |
| :--- | :--- | :--- |
| **Real-Time Voice Pipeline** | Asynchronous event loop with non-blocking session manager (`ConnectionManager`) supporting 100+ concurrent isolated audio streams and bidirectional WebSockets. | [`app/main.py`](app/main.py)<br>[`app/session.py`](app/session.py) |
| **Sub-500ms Response Times** | Pipeline overlapping: Deepgram streaming ASR ➔ Llama 3 token stream ➔ `SentenceBoundaryBuffer` ➔ Cartesia AI Sonic TTS streaming audio chunks before full completion finishes. | [`app/asr/deepgram_stream.py`](app/asr/deepgram_stream.py)<br>[`app/llm/llama_client.py`](app/llm/llama_client.py)<br>[`app/llm/sentence_buffer.py`](app/llm/sentence_buffer.py)<br>[`app/tts/cartesia_stream.py`](app/tts/cartesia_stream.py) |
| **Silero VAD & Voice Barge-In** | Embedded Silero VAD v5 ONNX runtime evaluating 32ms audio frames (16kHz). Detects speech onsets and triggers instant cancellation of in-flight LLM/TTS tasks + client audio buffer flushes with 95% accuracy. | [`app/vad/silero.py`](app/vad/silero.py)<br>[`app/session.py`](app/session.py)<br>[`static/app.js`](static/app.js) |
| **Adaptive Context Window** | Sliding 4-turn context window preventing prompt token buildup and eliminating latency drift over extended sessions. | [`app/session.py`](app/session.py) |

---

## 🏛️ System Architecture

```
User Mic (16kHz PCM) 
       │
       ▼
 [WebSocket Client] ──(Binary PCM Frames)──► [FastAPI WebSocket Endpoint]
                                                     │
                             ┌───────────────────────┴───────────────────────┐
                             ▼                                               ▼
                     [Silero VAD (ONNX)]                          [Deepgram Streaming ASR]
                             │                                               │
             (User speaks during playback?)                         (Interim / Final text)
                             │                                               │
                 [Barge-In Controller]                                       ▼
                             │                                        [Llama 3 (Groq)]
                 (Cancel LLM/TTS Tasks &                                     │
                  Flush Client Audio)                                 (Token Stream)
                             │                                               ▼
                             │                                [Sentence Boundary Buffer]
                             │                                               │
                             │                                   (Clause / Sentence Chunks)
                             │                                               ▼
                             │                                    [Cartesia Sonic TTS]
                             │                                               │
                             │                                        (Audio PCM Chunks)
                             ▼                                               ▼
 [WebSocket Client] ◄──(Interrupt Signal)────────────────────────────(Audio Packets)───
       │
       ▼
[Speaker Output] (Flushed on interrupt / Queued on speech)
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- Modern Web Browser (Chrome, Edge, Firefox, or Safari)

### 2. Installation
```powershell
# Clone the repository
git clone https://github.com/AnsFahad/Aura-Live-Streaming-Voice-Agent.git
cd Aura-Live-Streaming-Voice-Agent

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Copy `.env.example` to `.env`:
```powershell
cp .env.example .env
```

Configure your API keys in `.env`:
- **Groq API Key**: Ultra-fast LLM inference at [console.groq.com](https://console.groq.com) (free tier available).
- **Deepgram API Key**: Streaming speech-to-text at [console.deepgram.com](https://console.deepgram.com).
- **Cartesia AI API Key**: Streaming Sonic TTS at [play.cartesia.ai](https://play.cartesia.ai).

*(Note: If no API keys are provided, Aura automatically runs in **Offline Simulation Mode** with built-in Silero VAD and conversational intelligence).*

### 4. Running the Server
```powershell
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser at **`http://localhost:8000`**, click **"Start Microphone"**, and begin talking naturally!

---

## 🧪 Testing & Verification

### Unit Test Suite
Aura includes unit tests covering the Silero VAD ONNX engine, 16-bit PCM buffer conversion, and sentence boundary chunking:

```powershell
python -m pytest tests/test_pipeline.py -v
```

### 100-Session Concurrency Stress Benchmark
Simulate and benchmark 100 concurrent live audio streaming WebSocket connections:

```powershell
python scripts/benchmark_concurrency.py
```

---

## 💡 System Design & Technical Insights

### How Sentence Boundary Buffering (SBB) Achieves <500ms Latency
Standard conversational AI architectures are serialized: User speaks $\rightarrow$ ASR produces full text (500ms) $\rightarrow$ LLM produces full 30-word response (1,500ms) $\rightarrow$ TTS synthesizes full audio (1,000ms). Total response latency = **3,000ms+ (3 seconds)**.

Aura eliminates serialization by overlapping all three stages:
1. **Streaming ASR**: Deepgram streams partial tokens over WebSockets as the user is speaking.
2. **Fast Inference**: Llama 3 produces its first tokens within **~120ms (TTFT)**.
3. **Clause Chunking**: A sliding regex buffer identifies primary punctuation (`.`, `?`, `!`) or secondary clause boundaries (`,`, `;` after 5 words).
4. **Incremental TTS**: The moment the first clause is formed (e.g. 4 to 8 words), it is immediately dispatched to Cartesia's streaming WebSocket.
5. Cartesia synthesizes and streams audio packets within **150ms**.
By overlapping token generation with audio synthesis, the user hears audio in **under 450ms (TTFA)** while the LLM continues generating later tokens.

---

### Silero VAD & Voice-Activated Barge-In
Conversations feel unnatural if the user cannot interrupt the assistant. Aura embeds **Silero VAD v5** running locally on an **ONNX runtime engine** analyzing 32ms audio frames (512 samples at 16kHz):
1. **Real-time Evaluation**: Continuously assesses incoming microphone frames for speech probability.
2. **Instant Cancellation**: When speech is detected while the assistant is speaking, the session controller immediately issues `asyncio.Task.cancel()` on the LLM and TTS tasks.
3. **Client Buffer Flush**: Dispatches a WebSocket `{ "type": "interrupt" }` control packet, prompting the client's Web Audio pipeline to dump its playback buffer in under 10ms.
4. **History Truncation**: Truncates dialogue history to only the words that were actually voiced prior to interruption so conversation memory remains coherent.

---

### Asynchronous Scaling Across 100 Concurrent Sessions
The backend is built around Python's **`asyncio` event loop** and non-blocking WebSockets in FastAPI:
- **Lightweight Sessions**: Each user session is encapsulated in an isolated `SessionContext` object holding minimal state: an ONNX recurrent hidden state `[2, 1, 128]` and async queues.
- **CPU Efficiency**: Silero VAD inference on a 32ms frame takes under **0.4ms** of CPU time on modern cores.
- **Zero Thread Contention**: All network I/O (WebSocket frames, external API streaming) is purely asynchronous, using no OS-level blocking threads.
- **Memory Footprint**: Each active session consumes approximately **10–15 MB of RAM**, allowing 100 concurrent sessions to comfortably run within **1.5 GB of RAM**.

---

## 📁 Repository Structure

```text
Aura-Live-Streaming-Voice-Agent/
├── app/
│   ├── config.py                 # Pydantic configuration & API settings
│   ├── main.py                   # FastAPI app, connection manager (100 sessions), WebSockets
│   ├── session.py                # Pipeline orchestrator, barge-in controller, latency tracking
│   ├── vad/
│   │   ├── silero.py             # Silero VAD ONNX engine & PCM accumulator
│   │   └── models/
│   │       └── silero_vad.onnx   # Official Silero VAD v5 ONNX model
│   ├── asr/
│   │   └── deepgram_stream.py    # Deepgram streaming client + mock fallback
│   ├── llm/
│   │   ├── llama_client.py       # Llama 3 / Qwen streaming client + dynamic context
│   │   └── sentence_buffer.py    # Sentence boundary buffering logic
│   └── tts/
│       └── cartesia_stream.py    # Cartesia AI Sonic streaming client + fallback
├── static/
│   ├── index.html                # Modern interactive UI with visualizer & metrics
│   ├── style.css                 # Dark cyberpunk styling & animations
│   ├── app.js                    # Web Audio API, WebSocket client, queue & voice barge-in
│   └── pcm-processor.js          # AudioWorklet for 16kHz linear PCM mic capture
├── scripts/
│   └── benchmark_concurrency.py  # 100-session concurrency stress testing tool
├── tests/
│   └── test_pipeline.py          # Unit tests for VAD, buffering, and pipeline
├── requirements.txt
├── .env.example
└── README.md
```

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
