import os
import json
import logging
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv()

from voice.vad import SileroVAD, CHUNK_BYTES, SAMPLE_RATE
from voice.stt import FasterWhisperSTT
from voice.tts import KokoroTTS
from voice.agent import VoiceRentalAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice_server")

# Global models initialized on server startup
vad_model: Optional[SileroVAD] = None
stt_model: Optional[FasterWhisperSTT] = None
tts_model: Optional[KokoroTTS] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vad_model, stt_model, tts_model
    logger.info("Initializing $0 Voice Stack components...")
    vad_model = SileroVAD()
    stt_model = FasterWhisperSTT(model_size="base.en", device="cpu", compute_type="int8")
    tts_model = KokoroTTS(voice="af_heart", speed=1.05)
    logger.info("All Voice Stack models loaded successfully.")
    yield
    logger.info("Voice server shut down.")


app = FastAPI(title="HabitatOS $0 Voice Agent", lifespan=lifespan)

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class VoiceSession:
    """
    Manages a single real-time voice session over WebSocket:
    - Buffers incoming 16kHz PCM audio chunks.
    - Runs local Silero VAD (ONNX) in <2ms to detect speech onset.
    - Handles Instant Barge-In interruption (cancels active TTS & LLM task).
    - Detects end-of-speech after ~450ms silence.
    - Transcribes via Faster-Whisper.
    - Queries 4-Stage Postgres Hybrid RAG + LambdaMART + Quantile Valuation.
    - Streams generated 24kHz Kokoro TTS audio back to client.
    """

    def __init__(self, ws: WebSocket, vad: SileroVAD, stt: FasterWhisperSTT, tts: KokoroTTS):
        self.ws = ws
        self.vad = vad
        self.stt = stt
        self.tts = tts
        self.agent = VoiceRentalAgent()

        self.audio_buffer = bytearray()
        self.incoming_window = bytearray()
        self.is_speaking = False
        self.silence_chunks = 0
        self.bot_task: Optional[asyncio.Task] = None

    async def interrupt_bot(self):
        """
        Instant barge-in: If assistant is currently speaking or computing,
        cancel the active asyncio task and notify the client to clear audio playback.
        """
        if self.bot_task and not self.bot_task.done():
            logger.info("Barge-in triggered: Interrupting bot speech generation.")
            self.bot_task.cancel()
            try:
                await self.ws.send_text(json.dumps({"type": "clear_audio"}))
            except Exception as e:
                logger.warning(f"Error sending clear_audio on barge-in: {e}")

    async def process_incoming_audio(self, pcm_bytes: bytes):
        """
        Stream incoming PCM audio bytes (chunks of any size, typically 20-32ms).
        Window them into 512-sample (1024-byte) frames for Silero VAD.
        """
        self.incoming_window.extend(pcm_bytes)

        while len(self.incoming_window) >= CHUNK_BYTES:
            chunk = bytes(self.incoming_window[:CHUNK_BYTES])
            del self.incoming_window[:CHUNK_BYTES]
            await self._process_vad_frame(chunk)

    async def _process_vad_frame(self, chunk: bytes):
        speech_prob = self.vad.get_speech_probability(chunk)

        if speech_prob > 0.65:
            # User speaking detected
            if not self.is_speaking:
                logger.info(f"Speech onset detected (prob={speech_prob:.2f})")
                await self.interrupt_bot()
                self.is_speaking = True
                await self.ws.send_text(json.dumps({"type": "vad_state", "state": "speaking"}))

            self.silence_chunks = 0
            self.audio_buffer.extend(chunk)

        elif self.is_speaking:
            # User was speaking, now silent chunk
            self.audio_buffer.extend(chunk)
            self.silence_chunks += 1

            # ~450ms silence (14 chunks * 32ms) -> End of User Speech Turn
            if self.silence_chunks >= 14:
                logger.info("End of speech detected (silence duration reached).")
                utterance = bytes(self.audio_buffer)
                self.audio_buffer.clear()
                self.is_speaking = False
                self.silence_chunks = 0
                await self.ws.send_text(json.dumps({"type": "vad_state", "state": "processing"}))

                # Launch async response generation task
                self.bot_task = asyncio.create_task(self.respond(utterance))

    async def respond_text(self, user_text: str):
        """Execute response pipeline directly for text inputs or prompt chips."""
        try:
            await self.ws.send_text(json.dumps({"type": "vad_state", "state": "processing"}))
            agent_result = await self.agent.process_utterance(user_text)

            await self.ws.send_text(json.dumps({
                "type": "transcript",
                "role": "assistant",
                "text": agent_result["display_text"],
            }))

            if agent_result.get("listings"):
                await self.ws.send_text(json.dumps({
                    "type": "listings",
                    "data": agent_result["listings"],
                }))

            speech_text = agent_result.get("speech_text", "")
            sentences = self.tts.split_sentences(speech_text)

            await self.ws.send_text(json.dumps({"type": "vad_state", "state": "responding"}))

            for sentence in sentences:
                if asyncio.current_task().cancelled():
                    break

                pcm_audio = await self.tts.synthesize_sentence(sentence)
                if pcm_audio:
                    await self.ws.send_text(json.dumps({
                        "type": "audio_chunk",
                        "sample_rate": 24000,
                        "text": sentence,
                    }))
                    await self.ws.send_bytes(pcm_audio)
                    await asyncio.sleep(0.01)

            await self.ws.send_text(json.dumps({"type": "vad_state", "state": "listening"}))
        except asyncio.CancelledError:
            await self.ws.send_text(json.dumps({"type": "vad_state", "state": "listening"}))
        except Exception as e:
            logger.error(f"Error in respond_text: {e}")
            await self.ws.send_text(json.dumps({"type": "vad_state", "state": "listening"}))

    async def respond(self, pcm_bytes: bytes):
        """
        Full Turn-Taking Execution:
        1. Local Faster-Whisper STT
        2. HabitatOS 4-Stage Rental Retrieval & Valuation
        3. Kokoro-82M TTS Streaming
        """
        try:
            # 1. Faster-Whisper STT
            user_text = await self.stt.transcribe(pcm_bytes)
            if not user_text:
                logger.info("Whisper returned empty transcript. Resuming listening.")
                await self.ws.send_text(json.dumps({"type": "vad_state", "state": "listening"}))
                return

            logger.info(f"Transcribed User Query: '{user_text}'")
            await self.ws.send_text(json.dumps({
                "type": "transcript",
                "role": "user",
                "text": user_text,
            }))

            # 2. HabitatOS Rental Agent (4-stage pipeline)
            agent_result = await self.agent.process_utterance(user_text)

            await self.ws.send_text(json.dumps({
                "type": "transcript",
                "role": "assistant",
                "text": agent_result["display_text"],
            }))

            if agent_result.get("listings"):
                await self.ws.send_text(json.dumps({
                    "type": "listings",
                    "data": agent_result["listings"],
                }))

            # 3. Sentence Chunker + Kokoro TTS streaming
            speech_text = agent_result.get("speech_text", "")
            sentences = self.tts.split_sentences(speech_text)

            await self.ws.send_text(json.dumps({"type": "vad_state", "state": "responding"}))

            for sentence in sentences:
                if asyncio.current_task().cancelled():
                    logger.info("TTS streaming task cancelled by barge-in.")
                    break

                pcm_audio = await self.tts.synthesize_sentence(sentence)
                if pcm_audio:
                    # Notify browser audio header with 24kHz sample rate
                    await self.ws.send_text(json.dumps({
                        "type": "audio_chunk",
                        "sample_rate": 24000,
                        "text": sentence,
                    }))
                    # Stream raw 16-bit PCM bytes
                    await self.ws.send_bytes(pcm_audio)
                    # Small yield for smooth frame buffering
                    await asyncio.sleep(0.01)

            await self.ws.send_text(json.dumps({"type": "vad_state", "state": "listening"}))

        except asyncio.CancelledError:
            logger.info("Bot response cancelled due to user barge-in.")
            await self.ws.send_text(json.dumps({"type": "vad_state", "state": "listening"}))
        except Exception as e:
            logger.error(f"Error in response loop: {e}")
            await self.ws.send_text(json.dumps({
                "type": "transcript",
                "role": "assistant",
                "text": f"Error processing query: {e}",
            }))
            await self.ws.send_text(json.dumps({"type": "vad_state", "state": "listening"}))


@app.websocket("/ws/voice")
async def voice_websocket(ws: WebSocket):
    await ws.accept()
    logger.info("WebSocket voice client connected.")
    session = VoiceSession(ws, vad_model, stt_model, tts_model)

    try:
        await ws.send_text(json.dumps({
            "type": "system",
            "message": "Connected to HabitatOS $0 Voice Stack. Speak into your microphone.",
        }))
        while True:
            # Receive binary PCM audio or control text
            message = await ws.receive()
            if "bytes" in message and message["bytes"]:
                await session.process_incoming_audio(message["bytes"])
            elif "text" in message and message["text"]:
                try:
                    payload = json.loads(message["text"])
                    if payload.get("type") == "reset":
                        session.agent.reset_conversation()
                        await session.interrupt_bot()
                    elif payload.get("type") == "prompt" and payload.get("text"):
                        await session.interrupt_bot()
                        session.bot_task = asyncio.create_task(session.respond_text(payload["text"]))
                except json.JSONDecodeError:
                    pass
    except WebSocketDisconnect:
        logger.info("WebSocket voice client disconnected.")
        await session.interrupt_bot()
    except Exception as e:
        logger.error(f"WebSocket session exception: {e}")
        await session.interrupt_bot()


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h1>HabitatOS Voice Server Running</h1><p>static/index.html not found</p>")


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "vad_ready": vad_model is not None,
        "stt_ready": stt_model is not None,
        "tts_ready": tts_model is not None and tts_model.is_ready(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("voice.server:app", host="0.0.0.0", port=8001, reload=False)
