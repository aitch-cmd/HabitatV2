import logging
import asyncio
import numpy as np
from faster_whisper import WhisperModel

logger = logging.getLogger("voice_stt")


class FasterWhisperSTT:
    """
    Local CTranslate2 quantized Whisper STT running on CPU.
    Transcribes a 3-second utterance in ~180-250ms with zero API fees.
    """

    def __init__(self, model_size: str = "base.en", device: str = "cpu", compute_type: str = "int8"):
        self.model_size = model_size
        logger.info(f"Loading Faster-Whisper ({model_size}, {device}, {compute_type})...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        logger.info("Faster-Whisper STT model ready.")

    def _transcribe_sync(self, pcm_bytes: bytes) -> str:
        """Synchronous transcription of 16kHz 16-bit mono PCM."""
        if not pcm_bytes or len(pcm_bytes) < 1600:  # Less than 50ms of audio
            return ""

        audio_np = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        try:
            segments, info = self.model.transcribe(
                audio_np,
                beam_size=1,
                best_of=1,
                temperature=0.0,
                condition_on_previous_text=False,
                vad_filter=False,  # Already filtered by Silero VAD
                initial_prompt=(
                    "HabitatOS real estate voice assistant for Indian rental search: "
                    "1BHK, 2BHK, 3BHK, 4BHK, 1 BHK, 2 BHK, 3 BHK, flat, apartment, "
                    "under, budget, thousand, lakh, owner, direct from owner, "
                    "Rajiv Chowk, Connaught Place, Hauz Khas, Saket, Greater Kailash, "
                    "Vasant Kunj, Rohini, Janakpuri, Karampura, Dwarka, Mayur Vihar, "
                    "Noida, Gurgaon, Gurugram, South Delhi, Delhi, Mumbai, Bangalore."
                ),
            )
            text = " ".join(s.text.strip() for s in segments if s.text).strip()
            return text
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""

    async def transcribe(self, pcm_bytes: bytes) -> str:
        """Asynchronous non-blocking transcription via asyncio threadpool."""
        return await asyncio.to_thread(self._transcribe_sync, pcm_bytes)
