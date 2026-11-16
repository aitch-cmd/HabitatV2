import os
import re
import asyncio
import logging
from pathlib import Path
from typing import AsyncGenerator
import numpy as np

logger = logging.getLogger("voice_tts")

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "kokoro" / "kokoro-v1.0.onnx"
DEFAULT_VOICES_PATH = Path(__file__).resolve().parent.parent / "models" / "kokoro" / "voices-v1.0.bin"


class KokoroTTS:
    """
    Local Kokoro-82M ONNX Text-to-Speech Engine.
    Generates natural, human-sounding 24kHz PCM audio on CPU with ~320MB RAM.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        voices_path: str | Path = DEFAULT_VOICES_PATH,
        voice: str = "af_heart",
        speed: float = 1.05,
    ):
        self.model_path = Path(model_path)
        self.voices_path = Path(voices_path)
        self.voice = voice
        self.speed = speed
        self.kokoro = None
        self._init_engine()

    def _init_engine(self):
        if not self.model_path.exists() or not self.voices_path.exists():
            logger.warning(
                f"Kokoro model or voices files not found at {self.model_path}. "
                "Ensure kokoro-v1.0.onnx and voices-v1.0.bin are downloaded."
            )
            return

        try:
            from kokoro_onnx import Kokoro
            logger.info("Initializing Kokoro-82M ONNX engine...")
            self.kokoro = Kokoro(str(self.model_path), str(self.voices_path))
            logger.info("Kokoro-82M TTS engine ready.")
        except Exception as e:
            logger.error(f"Failed to initialize Kokoro ONNX: {e}")
            self.kokoro = None

    def is_ready(self) -> bool:
        if self.kokoro is None and self.model_path.exists() and self.voices_path.exists():
            self._init_engine()
        return self.kokoro is not None

    def _synthesize_sync(self, text: str) -> bytes:
        """Synchronously generate 24kHz 16-bit PCM bytes."""
        if not self.is_ready():
            logger.warning("Kokoro engine not ready; skipping audio generation.")
            return b""

        text = text.strip()
        if not text:
            return b""

        # Clean markdown characters for TTS reading
        clean_text = re.sub(r"[*_#`~\[\]]", "", text)
        clean_text = clean_text.replace("₹", " rupees ").replace("BHK", " B H K ")

        try:
            samples, sample_rate = self.kokoro.create(
                clean_text,
                voice=self.voice,
                speed=self.speed,
                lang="en-us",
            )
            # Convert float32 [-1, 1] to 16-bit signed PCM
            pcm_data = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
            return pcm_data
        except Exception as e:
            logger.error(f"TTS synthesis error for '{text[:40]}...': {e}")
            return b""

    async def synthesize_sentence(self, sentence: str) -> bytes:
        """Synthesize a single sentence asynchronously."""
        return await asyncio.to_thread(self._synthesize_sync, sentence)

    @staticmethod
    def split_sentences(text: str) -> list[str]:
        """Split text stream into sentences on punctuation boundaries for fast TTFA."""
        # Split on sentence terminals while keeping punctuation
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if s.strip()]
