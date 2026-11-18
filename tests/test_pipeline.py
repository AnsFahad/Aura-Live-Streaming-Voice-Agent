import pytest
import numpy as np
import os
from app.config import settings
from app.vad.silero import SileroVAD, VoiceActivityDetector
from app.llm.sentence_buffer import SentenceBoundaryBuffer

def test_silero_vad_initialization():
    assert os.path.exists(settings.VAD_MODEL_PATH), "VAD ONNX model file must exist"
    vad = SileroVAD(settings.VAD_MODEL_PATH, sample_rate=16000)
    
    # Silence frame (zeros)
    silence = np.zeros(512, dtype=np.float32)
    prob_silence = vad.process_frame(silence)
    assert 0.0 <= prob_silence < 0.1, f"Expected near zero probability for silence, got {prob_silence}"

def test_voice_activity_detector_pcm():
    detector = VoiceActivityDetector(
        model_path=settings.VAD_MODEL_PATH,
        threshold=0.5,
        sample_rate=16000,
        frame_size=512
    )
    
    # 512 samples of 16-bit zeros = 1024 bytes
    silence_pcm = bytes(1024)
    prob, turn_completed, should_barge_in = detector.process_pcm_bytes(silence_pcm)
    assert prob is not None
    assert prob < 0.1
    assert not turn_completed
    assert not should_barge_in

def test_sentence_boundary_buffering_primary():
    sbb = SentenceBoundaryBuffer()
    
    tokens = ["Hello ", "there! ", "How ", "can ", "I ", "help ", "you ", "today? ", "I ", "am ", "ready."]
    all_chunks = []
    
    for t in tokens:
        chunks = sbb.process_token(t)
        all_chunks.extend(chunks)
        
    all_chunks.extend(sbb.flush())
    
    assert len(all_chunks) == 3
    assert all_chunks[0] == "Hello there!"
    assert all_chunks[1] == "How can I help you today?"
    assert all_chunks[2] == "I am ready."

def test_sentence_boundary_buffering_clause_threshold():
    sbb = SentenceBoundaryBuffer(min_clause_words=3, max_buffer_words=6)
    
    # Long run without period, but with comma
    tokens = ["This ", "is ", "a ", "long ", "clause, ", "and ", "now ", "it ", "continues."]
    all_chunks = []
    for t in tokens:
        chunks = sbb.process_token(t)
        all_chunks.extend(chunks)
    all_chunks.extend(sbb.flush())

    assert len(all_chunks) >= 2
    assert "This is a long clause," in all_chunks[0]
