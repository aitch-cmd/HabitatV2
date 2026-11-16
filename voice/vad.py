import logging
import numpy as np
import torch

logger = logging.getLogger("voice_vad")

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512  # 32ms at 16kHz for Silero VAD
CHUNK_BYTES = CHUNK_SAMPLES * 2  # 16-bit PCM (2 bytes per sample)


class SileroVAD:
    """
    Local CPU-based Silero VAD (ONNX runtime)
    Detects speech onset in < 2ms per 32ms frame with zero external API calls.
    """

    def __init__(self, threshold: float = 0.65):
        self.threshold = threshold
        logger.info("Initializing Silero VAD ONNX model...")
        try:
            self.model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                onnx=True,
                trust_repo=True,
            )
            logger.info("Silero VAD ONNX model ready.")
        except Exception as e:
            logger.error(f"Failed to load Silero VAD: {e}")
            raise e

    def reset_states(self):
        """Reset internal recurrent state if available."""
        if hasattr(self.model, "reset_states"):
            self.model.reset_states()

    def get_speech_probability(self, pcm_bytes: bytes) -> float:
        """
        Evaluate speech probability for a 512-sample (1024-byte) 16kHz PCM chunk.
        """
        if len(pcm_bytes) != CHUNK_BYTES:
            return 0.0

        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        tensor = torch.from_numpy(samples)
        try:
            prob = self.model(tensor, SAMPLE_RATE).item()
            return float(prob)
        except Exception as e:
            logger.error(f"Error computing VAD score: {e}")
            return 0.0

    def is_speech(self, pcm_bytes: bytes) -> bool:
        return self.get_speech_probability(pcm_bytes) >= self.threshold
