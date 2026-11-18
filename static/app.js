// Aura Voice Agent Web Client
class AuraVoiceClient {
    constructor() {
        this.ws = null;
        this.audioContext = null;
        this.micStream = null;
        this.workletNode = null;
        this.scriptProcessor = null;
        this.zeroGain = null;
        this.recognition = null;
        this.recognitionRunning = false;
        this.isRecording = false;

        // Audio playback & speech queue state
        this.activeSourceNodes = [];
        this.isPlayingAudio = false;
        this.nextPlaybackTime = 0;
        this.isMockMode = true;
        this.speechQueue = [];
        this.isSpeakingQueueActive = false;
        this.selectedVoice = null;
        this.currentUtterance = null; // Strong reference to prevent V8 GC bug
        this.audioMode = "headphones";

        // Keep-alive intervals
        this.heartbeatInterval = null;
        this.speechUnfreezeInterval = null;

        // DOM Elements
        this.orb = document.getElementById("orb");
        this.statusLabel = document.getElementById("statusLabel");
        this.btnToggle = document.getElementById("btnToggle");
        this.voiceSelect = document.getElementById("voiceSelect");
        this.audioModeSelect = document.getElementById("audioModeSelect");
        this.connDot = document.getElementById("connDot");
        this.connStatus = document.getElementById("connStatus");
        this.chatHistory = document.getElementById("chatHistory");
        this.textInput = document.getElementById("textInput");
        this.btnSend = document.getElementById("btnSend");
        this.vadBar = document.getElementById("vadBar");
        this.valTtfa = document.getElementById("valTtfa");
        this.valTtft = document.getElementById("valTtft");
        this.valBargeIn = document.getElementById("valBargeIn");
        this.valMode = document.getElementById("valMode");

        this.currentAssistantBubble = null;
        this.currentUserBubble = null;

        this.initEventListeners();
        this.initVoiceSelector();
        this.initSpeechRecognition();
        this.connectWebSocket();
        this.startKeepAliveWatchdog();
    }

    initEventListeners() {
        this.btnToggle.addEventListener("click", () => this.toggleMicrophone());
        this.orb.addEventListener("click", () => this.toggleMicrophone());
        this.btnSend.addEventListener("click", () => this.sendTextMessage());
        this.textInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter") this.sendTextMessage();
        });
        if (this.audioModeSelect) {
            this.audioModeSelect.value = "headphones";
            this.audioModeSelect.addEventListener("change", (e) => {
                this.audioMode = e.target.value;
            });
        }
    }

    initVoiceSelector() {
        if (!("speechSynthesis" in window)) return;

        const populateVoices = () => {
            const voices = window.speechSynthesis.getVoices();
            if (!this.voiceSelect || voices.length === 0) return;

            this.voiceSelect.innerHTML = "";
            let bestVoice = null;

            voices.forEach((v) => {
                if (v.lang.startsWith("en")) {
                    const opt = document.createElement("option");
                    opt.value = v.name;
                    opt.textContent = `${v.name} (${v.lang})`;
                    this.voiceSelect.appendChild(opt);

                    if (!bestVoice && (v.name.includes("Natural") || v.name.includes("Google") || v.name.includes("Jenny") || v.name.includes("Aria") || v.name.includes("Samantha"))) {
                        bestVoice = v;
                    }
                }
            });

            if (bestVoice) {
                this.voiceSelect.value = bestVoice.name;
                this.selectedVoice = bestVoice;
            } else if (voices.length > 0) {
                this.selectedVoice = voices[0];
            }
        };

        populateVoices();
        if (speechSynthesis.onvoiceschanged !== undefined) {
            speechSynthesis.onvoiceschanged = populateVoices;
        }

        if (this.voiceSelect) {
            this.voiceSelect.addEventListener("change", () => {
                const voices = window.speechSynthesis.getVoices();
                this.selectedVoice = voices.find(v => v.name === this.voiceSelect.value) || null;
            });
        }
    }

    initSpeechRecognition() {
        const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRec) return;

        try {
            if (this.recognition) {
                this.recognition.onresult = null;
                this.recognition.onerror = null;
                this.recognition.onend = null;
                this.recognition.onstart = null;
                try { this.recognition.abort(); } catch (e) {}
            }
        } catch (e) {}

        this.recognition = new SpeechRec();
        this.recognition.continuous = true;
        this.recognition.interimResults = true;
        this.recognition.lang = "en-US";

        this.recognition.onstart = () => {
            this.recognitionRunning = true;
        };

        this.recognition.onresult = (event) => {
            let interim = "";
            let final = "";

            for (let i = event.resultIndex; i < event.results.length; ++i) {
                if (event.results[i].isFinal) {
                    final += event.results[i][0].transcript;
                } else {
                    interim += event.results[i][0].transcript;
                }
            }

            const spokenText = (final || interim).trim();

            // Automatic Voice Barge-in
            if (spokenText && (this.isPlayingAudio || ("speechSynthesis" in window && window.speechSynthesis.speaking))) {
                console.log("⚡ User speech detected -> Automatic Voice Barge-in!");
                this.flushAudioPlayback();
                this.handleBargeInInterruption({ barge_in_count: parseInt(this.valBargeIn.textContent || "0") + 1 });
                if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                    this.ws.send(JSON.stringify({ type: "barge_in_interrupt" }));
                }
            }

            if (interim) {
                this.renderUserMessage(interim, false);
                if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                    this.ws.send(JSON.stringify({
                        type: "client_transcript",
                        text: interim,
                        is_final: false
                    }));
                }
            }

            if (final) {
                this.renderUserMessage(final, true);
                if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                    this.ws.send(JSON.stringify({
                        type: "client_transcript",
                        text: final,
                        is_final: true
                    }));
                }
            }
        };

        this.recognition.onerror = (e) => {
            if (e.error !== "no-speech" && e.error !== "aborted") {
                console.warn("Speech recognition notice:", e.error);
            }
        };

        this.recognition.onend = () => {
            this.recognitionRunning = false;
            // Immediate debounced resurrection to prevent Chrome silence stalls
            if (this.isRecording) {
                setTimeout(() => {
                    this.safeStartRecognition();
                }, 100);
            }
        };
    }

    safeStartRecognition() {
        if (!this.isRecording || !this.recognition || this.recognitionRunning) return;
        try {
            this.recognition.start();
        } catch (e) {
            // If already started or transitioning, re-instantiate cleanly
            try {
                this.initSpeechRecognition();
                this.recognition.start();
            } catch (err) {}
        }
    }

    startKeepAliveWatchdog() {
        // 1. WebSocket keep-alive ping every 10s
        setInterval(() => {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({ type: "ping" }));
            }
        }, 10000);

        // 2. Speech recognition watchdog: Ensure recognition never dies after 5-6 sentences
        setInterval(() => {
            if (this.isRecording && !this.isPlayingAudio && !this.recognitionRunning) {
                this.safeStartRecognition();
            }
        }, 2000);

        // 3. SpeechSynthesis unfreeze timer (prevents Chrome 15s engine freeze)
        setInterval(() => {
            if (window.speechSynthesis && window.speechSynthesis.speaking) {
                window.speechSynthesis.pause();
                window.speechSynthesis.resume();
            }
        }, 3000);
    }

    connectWebSocket() {
        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        const wsUrl = `${protocol}//${window.location.host}/ws/chat`;

        this.ws = new WebSocket(wsUrl);
        this.ws.binaryType = "arraybuffer";

        this.ws.onopen = () => {
            this.connDot.classList.add("connected");
            this.connStatus.textContent = "Connected";
            this.statusLabel.textContent = "Ready - Click to Talk";
        };

        this.ws.onmessage = async (event) => {
            if (event.data instanceof ArrayBuffer) {
                this.queueAudioChunk(event.data);
            } else {
                try {
                    const msg = JSON.parse(event.data);
                    this.handleControlMessage(msg);
                } catch (e) {
                    console.error("Failed to parse control frame:", e);
                }
            }
        };

        this.ws.onclose = () => {
            this.connDot.classList.remove("connected");
            this.connStatus.textContent = "Disconnected";
            this.statusLabel.textContent = "Reconnecting...";
            setTimeout(() => this.connectWebSocket(), 2000);
        };
    }

    handleControlMessage(msg) {
        switch (msg.type) {
            case "connected":
                if (this.valMode) {
                    const modes = msg.modes || {};
                    this.isMockMode = (modes.tts_mode === "mock");
                    this.valMode.textContent = this.isMockMode ? "Simulation" : "Production";
                }
                break;

            case "vad_activity":
                const pct = Math.min(100, Math.round(msg.speech_prob * 100));
                if (this.vadBar) this.vadBar.style.width = `${pct}%`;
                if (msg.is_speaking && !this.isPlayingAudio) {
                    this.statusLabel.textContent = "Listening to you...";
                }
                break;

            case "user_transcript":
                this.renderUserMessage(msg.text, msg.is_final);
                break;

            case "assistant_transcript":
                this.renderAssistantMessage(msg.text, msg.is_final);
                break;

            case "speak_sentence":
                this.speakNaturalVoice(msg.text);
                break;

            case "metric_ttfa":
                if (this.valTtfa) {
                    this.valTtfa.textContent = `${msg.ttfa_ms}ms`;
                }
                break;

            case "metric_ttft":
                if (this.valTtft) {
                    this.valTtft.textContent = `${msg.ttft_ms}ms`;
                }
                break;

            case "interrupt":
                this.handleBargeInInterruption(msg);
                break;
        }
    }

    speakNaturalVoice(text) {
        if (!("speechSynthesis" in window)) return;

        const clean = text.replace(/[*#_`]/g, "").trim();
        if (!clean) return;

        this.speechQueue.push(clean);
        this.processSpeechQueue();
    }

    processSpeechQueue() {
        if (this.isSpeakingQueueActive || this.speechQueue.length === 0) return;

        const text = this.speechQueue.shift();
        this.isSpeakingQueueActive = true;
        this.isPlayingAudio = true;
        this.orb.classList.add("speaking");
        this.statusLabel.textContent = "Aura speaking... (Speak to interrupt)";

        if ("speechSynthesis" in window) {
            window.speechSynthesis.resume();
        }

        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 1.15;
        utterance.pitch = 1.0;

        // Store strong root reference to prevent Chromium Bug #679437 garbage-collection freeze!
        this.currentUtterance = utterance;
        window._activeUtterance = utterance;

        if (this.selectedVoice) {
            utterance.voice = this.selectedVoice;
        } else {
            const voices = window.speechSynthesis.getVoices();
            const preferred = voices.find(v =>
                v.lang.startsWith("en") &&
                (v.name.includes("Natural") || v.name.includes("Google") || v.name.includes("Microsoft") || v.name.includes("Jenny"))
            );
            if (preferred) utterance.voice = preferred;
        }

        let completed = false;
        const completeUtterance = () => {
            if (completed) return;
            completed = true;
            this.isSpeakingQueueActive = false;
            this.currentUtterance = null;
            window._activeUtterance = null;

            if (this.speechQueue.length > 0) {
                this.processSpeechQueue();
            } else {
                this.isPlayingAudio = false;
                this.orb.classList.remove("speaking");
                this.statusLabel.textContent = this.isRecording ? "Listening... Speak naturally" : "Ready";
                if (this.isRecording) {
                    this.safeStartRecognition();
                }
            }
        };

        utterance.onend = completeUtterance;
        utterance.onerror = completeUtterance;

        // Failsafe watchdog timer (5s max per sentence)
        setTimeout(() => {
            if (!completed) completeUtterance();
        }, 5000);

        window.speechSynthesis.speak(utterance);
    }

    handleBargeInInterruption(msg) {
        console.warn("⚡ AUTOMATIC BARGE-IN: User speech interrupted playback!");
        this.flushAudioPlayback();

        this.orb.classList.remove("speaking");
        this.orb.classList.add("interrupted");
        this.statusLabel.textContent = "Interrupted! Listening to your voice...";
        setTimeout(() => this.orb.classList.remove("interrupted"), 400);

        if (this.valBargeIn && msg.barge_in_count) {
            this.valBargeIn.textContent = msg.barge_in_count;
        }

        if (this.isRecording) {
            this.safeStartRecognition();
        }
    }

    async toggleMicrophone() {
        if (this.isRecording) {
            this.stopMicrophone();
        } else {
            await this.startMicrophone();
        }
    }

    async startMicrophone() {
        try {
            if ("speechSynthesis" in window) {
                window.speechSynthesis.resume();
            }

            if (!this.audioContext) {
                this.audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
            }
            if (this.audioContext.state === "suspended") {
                await this.audioContext.resume();
            }

            this.micStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    sampleRate: 16000,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });

            const source = this.audioContext.createMediaStreamSource(this.micStream);

            this.zeroGain = this.audioContext.createGain();
            this.zeroGain.gain.value = 0.0;

            try {
                await this.audioContext.audioWorklet.addModule("/static/pcm-processor.js");
                this.workletNode = new AudioWorkletNode(this.audioContext, "pcm-processor");
                this.workletNode.port.onmessage = (e) => {
                    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                        this.ws.send(e.data);
                    }
                };
                source.connect(this.workletNode);
            } catch (workletErr) {
                this.scriptProcessor = this.audioContext.createScriptProcessor(512, 1, 1);
                this.scriptProcessor.onaudioprocess = (e) => {
                    const inputData = e.inputBuffer.getChannelData(0);
                    const pcmData = new Int16Array(inputData.length);
                    for (let i = 0; i < inputData.length; i++) {
                        const s = Math.max(-1, Math.min(1, inputData[i]));
                        pcmData[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                    }
                    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                        this.ws.send(pcmData.buffer);
                    }
                };
                source.connect(this.scriptProcessor);
                this.scriptProcessor.connect(this.zeroGain);
                this.zeroGain.connect(this.audioContext.destination);
            }

            this.isRecording = true;
            this.safeStartRecognition();

            this.btnToggle.textContent = "Stop Microphone";
            this.btnToggle.classList.remove("btn-primary");
            this.btnToggle.classList.add("btn-danger");
            this.statusLabel.textContent = "Listening... Speak naturally";
        } catch (err) {
            console.error("Microphone access error:", err);
            alert("Microphone permission denied or device not found: " + err.message);
        }
    }

    stopMicrophone() {
        if (this.micStream) {
            this.micStream.getTracks().forEach(track => track.stop());
            this.micStream = null;
        }
        if (this.workletNode) {
            this.workletNode.disconnect();
            this.workletNode = null;
        }
        if (this.scriptProcessor) {
            this.scriptProcessor.disconnect();
            this.scriptProcessor = null;
        }
        if (this.recognition) {
            try { this.recognition.stop(); } catch (e) {}
        }
        this.isRecording = false;
        this.recognitionRunning = false;
        this.btnToggle.textContent = "Start Microphone";
        this.btnToggle.classList.remove("btn-danger");
        this.btnToggle.classList.add("btn-primary");
        this.statusLabel.textContent = "Microphone paused";
    }

    queueAudioChunk(arrayBuffer) {
        if (!this.audioContext) {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        }

        const int16 = new Int16Array(arrayBuffer);
        const float32 = new Float32Array(int16.length);
        for (let i = 0; i < int16.length; i++) {
            float32[i] = int16[i] / 32768.0;
        }

        const audioBuffer = this.audioContext.createBuffer(1, float32.length, 16000);
        audioBuffer.getChannelData(0).set(float32);

        this.playBuffer(audioBuffer);
    }

    playBuffer(audioBuffer) {
        if (this.audioContext.state === "suspended") {
            this.audioContext.resume();
        }

        const source = this.audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(this.audioContext.destination);

        const currentTime = this.audioContext.currentTime;
        const startTime = Math.max(currentTime, this.nextPlaybackTime);
        source.start(startTime);
        this.nextPlaybackTime = startTime + audioBuffer.duration;

        this.activeSourceNodes.push(source);
        this.isPlayingAudio = true;
        this.orb.classList.add("speaking");
        this.statusLabel.textContent = "Aura speaking...";

        source.onended = () => {
            const idx = this.activeSourceNodes.indexOf(source);
            if (idx > -1) this.activeSourceNodes.splice(idx, 1);
            if (this.activeSourceNodes.length === 0) {
                this.isPlayingAudio = false;
                this.orb.classList.remove("speaking");
                this.statusLabel.textContent = this.isRecording ? "Listening..." : "Ready";
            }
        };
    }

    flushAudioPlayback() {
        this.speechQueue = [];
        this.isSpeakingQueueActive = false;
        this.currentUtterance = null;
        window._activeUtterance = null;

        if ("speechSynthesis" in window) {
            window.speechSynthesis.cancel();
        }

        for (const source of this.activeSourceNodes) {
            try {
                source.stop();
                source.disconnect();
            } catch (e) {}
        }
        this.activeSourceNodes = [];
        this.nextPlaybackTime = 0;
        this.isPlayingAudio = false;
    }

    renderUserMessage(text, isFinal) {
        if (!this.currentUserBubble) {
            const row = document.createElement("div");
            row.className = "message-row";
            row.innerHTML = `
                <div class="message-role user">You</div>
                <div class="message-bubble ${isFinal ? '' : 'partial'}"></div>
            `;
            this.chatHistory.appendChild(row);
            this.currentUserBubble = row.querySelector(".message-bubble");
        }

        this.currentUserBubble.textContent = text;
        if (isFinal) {
            this.currentUserBubble.classList.remove("partial");
            this.currentUserBubble = null;
        }
        this.chatHistory.scrollTop = this.chatHistory.scrollHeight;
    }

    renderAssistantMessage(text, isFinal) {
        if (!this.currentAssistantBubble) {
            const row = document.createElement("div");
            row.className = "message-row";
            row.innerHTML = `
                <div class="message-role assistant">Aura</div>
                <div class="message-bubble ${isFinal ? '' : 'partial'}"></div>
            `;
            this.chatHistory.appendChild(row);
            this.currentAssistantBubble = row.querySelector(".message-bubble");
        }

        this.currentAssistantBubble.textContent = text;
        if (isFinal) {
            this.currentAssistantBubble.classList.remove("partial");
            this.currentAssistantBubble = null;
        }
        this.chatHistory.scrollTop = this.chatHistory.scrollHeight;
    }

    sendTextMessage() {
        const text = this.textInput.value.trim();
        if (!text) return;

        this.renderUserMessage(text, true);
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: "text_prompt", text: text }));
        }
        this.textInput.value = "";
    }
}

// Initialize on page load
window.addEventListener("DOMContentLoaded", () => {
    window.auraClient = new AuraVoiceClient();
});
