import os
import numpy as np
import onnxruntime as ort
from typing import Tuple, Optional
import logging

logger = logging.getLogger("aura.vad")

class SileroVAD:
    """
    Direct Silero VAD v5 ONNX engine.
    Processes 16kHz audio in 512-sample (32ms) windows and tracks recurrent hidden states.
    """
    def __init__(self, model_path: str, sample_rate: int = 16000):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Silero VAD ONNX model not found at {model_path}")
        
        # Configure ONNX session with minimal latency options
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(model_path, sess_options=opts, providers=['CPUExecutionProvider'])
        self.sample_rate = sample_rate
        self.sr_array = np.array(sample_rate, dtype=np.int64)
        self.reset()
        logger.info(f"Silero VAD initialized from {model_path} at {sample_rate}Hz")

    def reset(self):
        """Reset the internal hidden states of the recurrent model."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)

    def process_frame(self, frame_float32: np.ndarray) -> float:
        """
        Process a single 512-sample frame of float32 audio [-1.0, 1.0].
        Returns speech probability between 0.0 and 1.0.
        """
        if len(frame_float32.shape) == 1:
            frame_float32 = np.expand_dims(frame_float32, axis=0)

        inputs = {
            'input': frame_float32.astype(np.float32),
            'state': self._state,
            'sr': self.sr_array
        }

        out, new_state = self.session.run(None, inputs)
        self._state = new_state
        return float(out[0][0])


class VoiceActivityDetector:
    """
    Stateful VAD accumulator and speech turn tracker.
    Handles raw 16-bit PCM byte stream, detects speech onsets/offsets,
    and flags barge-in interruptions.
    """
    def __init__(
        self,
        model_path: str,
        threshold: float = 0.5,
        sample_rate: int = 16000,
        frame_size: int = 512,
        silence_timeout_ms: int = 600,
        barge_in_consecutive_frames: int = 3
    ):
        self.vad = SileroVAD(model_path, sample_rate)
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.silence_timeout_ms = silence_timeout_ms
        self.barge_in_consecutive_frames = barge_in_consecutive_frames

        # Buffer for incoming audio bytes until we have full 512-sample frames
        self._byte_buffer = bytearray()
        self.bytes_per_frame = frame_size * 2  # 16-bit = 2 bytes per sample

        # State tracking
        self.is_speech_active = False
        self.consecutive_speech_frames = 0
        self.consecutive_silence_frames = 0
        self.silence_duration_ms = 0.0
        self.frame_duration_ms = (frame_size / sample_rate) * 1000.0  # 32ms

    def reset(self):
        self.vad.reset()
        self._byte_buffer.clear()
        self.is_speech_active = False
        self.consecutive_speech_frames = 0
        self.consecutive_silence_frames = 0
        self.silence_duration_ms = 0.0

    def process_pcm_bytes(self, pcm_bytes: bytes, assistant_is_speaking: bool = False) -> Tuple[Optional[float], bool, bool]:
        """
        Accepts arbitrary size of raw 16-bit mono PCM bytes.
        Returns:
            (last_speech_prob, is_turn_completed, should_barge_in)
        """
        self._byte_buffer.extend(pcm_bytes)
        last_prob = None
        turn_completed = False
        barge_in_triggered = False

        while len(self._byte_buffer) >= self.bytes_per_frame:
            frame_bytes = self._byte_buffer[:self.bytes_per_frame]
            del self._byte_buffer[:self.bytes_per_frame]

            # Convert 16-bit PCM bytes to normalized float32 [-1.0, 1.0]
            audio_int16 = np.frombuffer(frame_bytes, dtype=np.int16)
            audio_float32 = (audio_int16.astype(np.float32)) / 32768.0

            prob = self.vad.process_frame(audio_float32)
            last_prob = prob

            effective_threshold = 0.58 if assistant_is_speaking else self.threshold
            barge_in_needed_frames = max(2, self.barge_in_consecutive_frames)

            if prob >= effective_threshold:
                self.consecutive_speech_frames += 1
                self.consecutive_silence_frames = 0
                self.silence_duration_ms = 0.0

                if not self.is_speech_active and self.consecutive_speech_frames >= 2:
                    self.is_speech_active = True

                # Check barge-in: User speaks while assistant is generating/speaking audio
                if assistant_is_speaking and self.consecutive_speech_frames >= barge_in_needed_frames:
                    barge_in_triggered = True
            else:
                self.consecutive_silence_frames += 1
                self.consecutive_speech_frames = 0
                self.silence_duration_ms += self.frame_duration_ms

                # Check if user finished speaking (silence timeout reached)
                if self.is_speech_active and self.silence_duration_ms >= self.silence_timeout_ms:
                    self.is_speech_active = False
                    turn_completed = True

        return last_prob, turn_completed, barge_in_triggered
