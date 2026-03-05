"""Hugging Face Inference API conversation handler for Reachy Mini.

Fully local audio pipeline + cloud LLM inference via Hugging Face:
  Audio In → faster-whisper (STT) → HF Inference API (LLM + tools) → pyttsx3 (TTS) → Audio Out

No local GPU or model download required — only a Hugging Face token with access to the model.

Setup:
1. Get a token at https://huggingface.co/settings/tokens
2. Request access to meta-llama/Llama-3.1-8B-Instruct on HF.
3. Add to your .env:
       HF_TOKEN=hf_your_token_here
       MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
"""

import json
import base64
import asyncio
import logging
import tempfile
import os
from typing import Any, Final, Tuple, Optional
from datetime import datetime

import cv2
import numpy as np
import gradio as gr
import pyttsx3
import miniaudio
from huggingface_hub import InferenceClient
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item, audio_to_int16
from numpy.typing import NDArray
from scipy.signal import resample

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.prompts import get_session_instructions
from reachy_mini_conversation_app.tools.core_tools import (
    ToolDependencies,
    get_tool_specs,
    dispatch_tool_call,
)


logger = logging.getLogger(__name__)

HANDLER_SAMPLE_RATE: Final[int] = 24000
WHISPER_SAMPLE_RATE: Final[int] = 16000

# Voice-activity detection thresholds
SILENCE_RMS_THRESHOLD: Final[float] = 500.0
SILENCE_DURATION_S: Final[float] = 0.8
MIN_SPEECH_DURATION_S: Final[float] = 0.3

DEFAULT_MODEL: Final[str] = "meta-llama/Llama-3.1-8B-Instruct"


def _get_hf_token() -> str:
    """Resolve Hugging Face token from config or environment."""
    return getattr(config, "HF_TOKEN", None) or os.environ.get("HF_TOKEN", "")


def _get_model_name() -> str:
    """Resolve model name from config, falling back to default."""
    return getattr(config, "MODEL_NAME", None) or DEFAULT_MODEL


class HuggingFaceHandler(AsyncStreamHandler):
    """Conversation handler using HF Inference API (LLM), faster-whisper (STT), and pyttsx3 (TTS)."""

    def __init__(
        self,
        deps: ToolDependencies,
        gradio_mode: bool = False,
        instance_path: Optional[str] = None,
    ):
        super().__init__(
            expected_layout="mono",
            output_sample_rate=HANDLER_SAMPLE_RATE,
            input_sample_rate=HANDLER_SAMPLE_RATE,
        )

        self.deps = deps
        self.gradio_mode = gradio_mode
        self.instance_path = instance_path

        self.output_queue: asyncio.Queue = asyncio.Queue()

        # HF Inference client (initialized in start_up)
        self._hf_client: InferenceClient | None = None

        # Whisper STT model (initialized in start_up)
        self.whisper_model: Any = None

        # Conversation history (kept across turns)
        self._messages: list[dict[str, Any]] = []

        # Audio buffering for VAD + STT
        self._audio_buffer: list[NDArray[np.int16]] = []
        self._is_speaking: bool = False
        self._silence_frame_count: int = 0
        self._speech_frame_count: int = 0
        self._frame_duration: float = 0.0

        # TTS lock – prevents overlapping synthesis
        self._tts_lock: asyncio.Lock = asyncio.Lock()

        # TTS voice
        self._tts_voice: str = getattr(config, "TTS_VOICE", "")

        # Timing
        self.last_activity_time: float = 0.0
        self.start_time: float = 0.0
        self.is_idle_tool_call: bool = False

        # Lifecycle flags
        self._shutdown_requested: bool = False
        self._connected_event: asyncio.Event = asyncio.Event()

    def copy(self) -> "HuggingFaceHandler":
        """Create a copy of this handler."""
        return HuggingFaceHandler(self.deps, self.gradio_mode, self.instance_path)

    # ------------------------------------------------------------------ #
    # Startup & lifecycle
    # ------------------------------------------------------------------ #

    async def start_up(self) -> None:
        """Initialize STT, HF Inference client, and keep running until shutdown."""
        loop = asyncio.get_event_loop()
        self.last_activity_time = loop.time()
        self.start_time = loop.time()

        # 1. Hugging Face Inference client
        hf_token = _get_hf_token()
        model_name = _get_model_name()

        if not hf_token:
            logger.error("HF_TOKEN is not set. Requests to Hugging Face will fail.")
        else:
            logger.info("HF_TOKEN found. Using model: %s", model_name)

        self._hf_client = InferenceClient(
            model=model_name,
            token=hf_token or None,
        )

        # 2. faster-whisper STT
        stt_model = getattr(config, "STT_MODEL", "base")
        try:
            from faster_whisper import WhisperModel

            self.whisper_model = await loop.run_in_executor(
                None,
                lambda: WhisperModel(stt_model, device="auto", compute_type="int8"),
            )
            logger.info("Loaded faster-whisper model: %s", stt_model)
        except Exception as e:
            logger.error("Failed to load STT model '%s': %s", stt_model, e)
            logger.warning("Speech-to-text will be unavailable.")

        # 3. Conversation bootstrap
        self._messages = [{"role": "system", "content": get_session_instructions()}]
        self._tts_voice = getattr(config, "TTS_VOICE", "")

        self._connected_event.set()
        logger.info(
            "HuggingFaceHandler ready — model=%s  stt=%s  tts_voice=%r",
            model_name,
            stt_model,
            self._tts_voice,
        )

        # Keep handler alive until shutdown
        while not self._shutdown_requested:
            await asyncio.sleep(0.1)

    # ------------------------------------------------------------------ #
    # Personality / session management
    # ------------------------------------------------------------------ #

    async def apply_personality(self, profile: str | None) -> str:
        """Apply a new personality at runtime by updating the system prompt."""
        try:
            from reachy_mini_conversation_app.config import set_custom_profile

            set_custom_profile(profile)
            instructions = get_session_instructions()
            self._messages = [{"role": "system", "content": instructions}]
            logger.info("Applied personality: %s", profile or "built-in default")
            return "Applied personality. Active on next message."
        except Exception as e:
            logger.error("Error applying personality '%s': %s", profile, e)
            return f"Failed to apply personality: {e}"

    async def _restart_session(self) -> None:
        """Reset conversation history."""
        self._messages = [{"role": "system", "content": get_session_instructions()}]
        logger.info("Session reset.")

    # ------------------------------------------------------------------ #
    # Audio receive (microphone) → VAD
    # ------------------------------------------------------------------ #

    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        """Receive audio frame, run simple energy VAD, fire pipeline when done."""
        if self._shutdown_requested or self.whisper_model is None:
            return

        input_sample_rate, audio_frame = frame

        # Reshape to 1-D mono
        if audio_frame.ndim == 2:
            if audio_frame.shape[1] > audio_frame.shape[0]:
                audio_frame = audio_frame.T
            if audio_frame.shape[1] > 1:
                audio_frame = audio_frame[:, 0]
        audio_frame = audio_frame.flatten()

        # Resample to handler rate if necessary
        if input_sample_rate != HANDLER_SAMPLE_RATE:
            n_out = int(len(audio_frame) * HANDLER_SAMPLE_RATE / input_sample_rate)
            audio_frame = resample(audio_frame, n_out)

        audio_frame = audio_to_int16(audio_frame)

        # Track frame duration (seconds per frame)
        self._frame_duration = len(audio_frame) / HANDLER_SAMPLE_RATE

        # Simple energy-based VAD
        rms = float(np.sqrt(np.mean(audio_frame.astype(np.float32) ** 2)))

        if rms > SILENCE_RMS_THRESHOLD:
            if not self._is_speaking:
                self._is_speaking = True
                self._speech_frame_count = 0
                self._silence_frame_count = 0
                if self.deps.head_wobbler is not None:
                    self.deps.head_wobbler.reset()
                self.deps.movement_manager.set_listening(True)
                logger.debug("Speech started (RMS=%.0f)", rms)
            self._silence_frame_count = 0
            self._speech_frame_count += 1
            self._audio_buffer.append(audio_frame)
        else:
            if self._is_speaking:
                self._silence_frame_count += 1
                self._audio_buffer.append(audio_frame)

                silence_duration = self._silence_frame_count * self._frame_duration
                if silence_duration >= SILENCE_DURATION_S:
                    speech_duration = self._speech_frame_count * self._frame_duration
                    self.deps.movement_manager.set_listening(False)

                    captured = np.concatenate(self._audio_buffer)
                    self._audio_buffer = []
                    self._is_speaking = False
                    self._silence_frame_count = 0
                    self._speech_frame_count = 0

                    if speech_duration >= MIN_SPEECH_DURATION_S:
                        logger.debug("Speech ended (%.2fs), processing...", speech_duration)
                        asyncio.create_task(self._process_speech(captured))
                    else:
                        logger.debug("Speech too short (%.2fs), discarded", speech_duration)

    # ------------------------------------------------------------------ #
    # Speech processing pipeline
    # ------------------------------------------------------------------ #

    async def _process_speech(self, audio_data: NDArray[np.int16]) -> None:
        """Full pipeline: STT → LLM (with tools) → TTS."""
        try:
            text = await self._transcribe(audio_data)
            if not text:
                logger.debug("Empty transcription; skipping.")
                return

            logger.info("User: %s", text)
            await self.output_queue.put(AdditionalOutputs({"role": "user", "content": text}))

            self._messages.append({"role": "user", "content": text})
            response_text = await self._chat_with_tools()

            if response_text:
                logger.info("Assistant: %s", response_text)
                await self.output_queue.put(
                    AdditionalOutputs({"role": "assistant", "content": response_text})
                )
                await self._synthesize_speech(response_text)

        except Exception as e:
            logger.error("Speech processing error: %s", e, exc_info=True)
            await self.output_queue.put(
                AdditionalOutputs({"role": "assistant", "content": f"[error] {e}"})
            )

    # ------------------------------------------------------------------ #
    # STT
    # ------------------------------------------------------------------ #

    async def _transcribe(self, audio_data: NDArray[np.int16]) -> str:
        """Run faster-whisper on raw PCM int16 audio at HANDLER_SAMPLE_RATE."""
        float_audio = audio_data.astype(np.float32) / 32768.0
        n_out = int(len(float_audio) * WHISPER_SAMPLE_RATE / HANDLER_SAMPLE_RATE)
        whisper_audio = resample(float_audio, n_out).astype(np.float32)

        loop = asyncio.get_event_loop()

        def _run():
            segs, _ = self.whisper_model.transcribe(
                whisper_audio,
                beam_size=5,
                language="en",
                vad_filter=True,
            )
            return [s.text for s in segs]

        parts: list[str] = await loop.run_in_executor(None, _run)
        return " ".join(parts).strip()

    # ------------------------------------------------------------------ #
    # LLM + tool calling via Hugging Face Inference API
    # ------------------------------------------------------------------ #

    async def _chat_with_tools(self) -> str:
        """Send conversation to HF Inference API; handle tool calls; return final text."""
        if self._hf_client is None:
            return "Hugging Face client not initialized."

        hf_tools = self._build_hf_tools()
        loop = asyncio.get_event_loop()

        # Agentic loop: iterate until no more tool calls (safety cap = 10)
        for _iteration in range(10):
            try:
                response = await loop.run_in_executor(
                    None,
                    lambda: self._hf_client.chat.completions.create(  # type: ignore[union-attr]
                        messages=self._messages,
                        tools=hf_tools if hf_tools else None,
                        tool_choice="auto" if hf_tools else None,
                        max_tokens=512,
                    ),
                )
            except Exception as e:
                logger.error("HF Inference API chat error: %s", e, exc_info=True)
                return f"[LLM error] {e}"

            choice = response.choices[0]
            assistant_msg = choice.message
            tool_calls = assistant_msg.tool_calls or []

            if not tool_calls:
                # Final text response
                response_text: str = assistant_msg.content or ""
                if response_text:
                    self._messages.append({"role": "assistant", "content": response_text})
                return response_text

            # ---- process tool calls ----
            self._messages.append({
                "role": "assistant",
                "content": assistant_msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ],
            })

            for tc in tool_calls:
                tool_name: str = tc.function.name
                raw_args = tc.function.arguments
                tool_args_json = raw_args if isinstance(raw_args, str) else json.dumps(raw_args)

                try:
                    tool_result = await dispatch_tool_call(tool_name, tool_args_json, self.deps)
                    logger.debug("Tool '%s' result: %s", tool_name, tool_result)
                except Exception as e:
                    logger.error("Tool '%s' failed: %s", tool_name, e)
                    tool_result = {"error": str(e)}

                # Emit tool result to UI
                await self.output_queue.put(
                    AdditionalOutputs(
                        {
                            "role": "assistant",
                            "content": json.dumps(tool_result),
                            "metadata": {
                                "title": f"🛠️ Used tool {tool_name}",
                                "status": "done",
                            },
                        }
                    )
                )

                # Camera tool → show image in chat
                if tool_name == "camera" and "b64_im" in tool_result:
                    if self.deps.camera_worker is not None:
                        np_img = self.deps.camera_worker.get_latest_frame()
                        rgb_frame = (
                            cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB) if np_img is not None else None
                        )
                        await self.output_queue.put(
                            AdditionalOutputs({"role": "assistant", "content": gr.Image(value=rgb_frame)})
                        )

                # Append tool result to conversation
                self._messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(tool_result),
                })

            if self.is_idle_tool_call:
                self.is_idle_tool_call = False
                return ""

        logger.warning("Exited tool-call loop after safety cap.")
        return ""

    @staticmethod
    def _build_hf_tools() -> list[dict[str, Any]]:
        """Convert internal tool specs to OpenAI-compatible format for HF Inference API."""
        return [
            {
                "type": "function",
                "function": {
                    "name": spec["name"],
                    "description": spec.get("description", ""),
                    "parameters": spec.get(
                        "parameters", {"type": "object", "properties": {}}
                    ),
                },
            }
            for spec in get_tool_specs()
        ]

    # ------------------------------------------------------------------ #
    # TTS
    # ------------------------------------------------------------------ #

    async def _synthesize_speech(self, text: str) -> None:
        """Convert text to speech via pyttsx3, stream audio into output_queue."""
        if not text.strip():
            return

        async with self._tts_lock:
            loop = asyncio.get_event_loop()
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
            os.close(tmp_fd)

            try:
                def _tts_save() -> bool:
                    """Run pyttsx3 synchronously and save to tmp_path."""
                    try:
                        engine = pyttsx3.init()
                        engine.setProperty("rate", 165)
                        voices = engine.getProperty("voices") or []
                        tts_voice = self._tts_voice
                        if tts_voice:
                            for v in voices:
                                name_match = tts_voice.lower() in (v.name or "").lower()
                                id_match = tts_voice.lower() in (v.id or "").lower()
                                if name_match or id_match:
                                    engine.setProperty("voice", v.id)
                                    break
                        engine.save_to_file(text, tmp_path)
                        engine.runAndWait()
                        engine.stop()
                        return True
                    except Exception as inner_e:
                        logger.error("pyttsx3 internal error: %s", inner_e)
                        return False

                success = await loop.run_in_executor(None, _tts_save)

                if not success or not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
                    logger.error("TTS produced no output for: %.60s", text)
                    return

                decoded = await loop.run_in_executor(
                    None,
                    lambda: miniaudio.decode_file(
                        tmp_path,
                        output_format=miniaudio.SampleFormat.SIGNED16,
                        nchannels=1,
                        sample_rate=HANDLER_SAMPLE_RATE,
                    ),
                )
                samples = np.frombuffer(decoded.samples, dtype=np.int16)

                # Stream in ~100 ms chunks
                chunk_size = HANDLER_SAMPLE_RATE // 10
                for i in range(0, len(samples), chunk_size):
                    if self._shutdown_requested:
                        break
                    chunk = samples[i : i + chunk_size]
                    if self.deps.head_wobbler is not None:
                        self.deps.head_wobbler.feed(
                            base64.b64encode(chunk.tobytes()).decode("utf-8")
                        )
                    self.last_activity_time = asyncio.get_event_loop().time()
                    await self.output_queue.put((HANDLER_SAMPLE_RATE, chunk.reshape(1, -1)))

            except Exception as e:
                logger.error("TTS synthesis error: %s", e, exc_info=True)
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    # ------------------------------------------------------------------ #
    # Emit (speaker output)
    # ------------------------------------------------------------------ #

    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Emit audio frame / metadata to the fastrtc Stream."""
        idle_duration = asyncio.get_event_loop().time() - self.last_activity_time
        if idle_duration > 15.0 and self.deps.movement_manager.is_idle():
            try:
                await self.send_idle_signal(idle_duration)
            except Exception as e:
                logger.warning("Idle signal skipped: %s", e)
                return None
            self.last_activity_time = asyncio.get_event_loop().time()

        return await wait_for_item(self.output_queue)  # type: ignore[no-any-return]

    # ------------------------------------------------------------------ #
    # Idle behaviour
    # ------------------------------------------------------------------ #

    async def send_idle_signal(self, idle_duration: float) -> None:
        """Prompt the LLM to perform idle/creative behaviour via tool calls."""
        logger.debug("Sending idle signal (%.1fs idle)", idle_duration)
        self.is_idle_tool_call = True
        idle_msg = (
            f"[Idle time update: {self.format_timestamp()} - No activity for {idle_duration:.1f}s] "
            "You've been idle for a while. Feel free to get creative - dance, show an emotion, "
            "look around, do nothing, or just be yourself!"
        )
        self._messages.append({"role": "user", "content": idle_msg})
        response_text = await self._chat_with_tools()
        if response_text and not self.is_idle_tool_call:
            await self._synthesize_speech(response_text)

    # ------------------------------------------------------------------ #
    # Voices
    # ------------------------------------------------------------------ #

    async def get_available_voices(self) -> list[str]:
        """Return available pyttsx3 voice names."""
        try:
            loop = asyncio.get_event_loop()

            def _get():
                engine = pyttsx3.init()
                names = [v.name for v in (engine.getProperty("voices") or [])]
                engine.stop()
                return names

            return await loop.run_in_executor(None, _get)
        except Exception:
            return ["Default"]

    # ------------------------------------------------------------------ #
    # Shutdown
    # ------------------------------------------------------------------ #

    async def shutdown(self) -> None:
        """Gracefully shut down the handler."""
        self._shutdown_requested = True

        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        self._connected_event.clear()
        logger.info("HuggingFaceHandler shut down.")

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #

    def format_timestamp(self) -> str:
        """Return a formatted timestamp string."""
        loop_time = asyncio.get_event_loop().time()
        elapsed = loop_time - self.start_time
        dt = datetime.now()
        return f"[{dt.strftime('%Y-%m-%d %H:%M:%S')} | +{elapsed:.1f}s]"
