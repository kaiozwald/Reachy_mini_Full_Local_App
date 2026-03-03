"""Ollama-based conversation handler with local STT (faster-whisper) and TTS (edge-tts).

Replaces the previous OpenAI Realtime API handler with a fully local/self-hosted
pipeline:
  Audio In → faster-whisper (STT) → Ollama (LLM + tools) → pyttsx3 (TTS) → Audio Out
"""

import io
import json
import base64
import asyncio
import logging
from typing import Any, Final, Tuple, Literal, Optional
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import gradio as gr
import pyttsx3
import miniaudio
from ollama import AsyncClient as OllamaAsyncClient
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item, audio_to_int16
from numpy.typing import NDArray
from scipy.signal import resample

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.prompts import get_session_voice, get_session_instructions
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
SILENCE_DURATION_S: Final[float] = 0.8  # seconds of silence to end utterance
MIN_SPEECH_DURATION_S: Final[float] = 0.3  # discard very short bursts


class OllamaHandler(AsyncStreamHandler):
    """Conversation handler using Ollama (LLM), faster-whisper (STT), and edge-tts (TTS)."""

    def __init__(
        self,
        deps: ToolDependencies,
        gradio_mode: bool = False,
        instance_path: Optional[str] = None,
    ):
        """Initialize the handler."""
        super().__init__(
            expected_layout="mono",
            output_sample_rate=HANDLER_SAMPLE_RATE,
            input_sample_rate=HANDLER_SAMPLE_RATE,
        )

        self.deps = deps
        self.gradio_mode = gradio_mode
        self.instance_path = instance_path

        # Output queue (audio frames + AdditionalOutputs for chat UI)
        self.output_queue: "asyncio.Queue[Tuple[int, NDArray[np.int16]] | AdditionalOutputs]" = asyncio.Queue()

        # Clients (initialized in start_up)
        self.ollama_client: OllamaAsyncClient | None = None
        self.whisper_model: Any = None  # faster_whisper.WhisperModel

        # Conversation history
        self._messages: list[dict[str, Any]] = []

        # Audio buffering for VAD + STT
        self._audio_buffer: list[NDArray[np.int16]] = []
        self._is_speaking: bool = False
        self._silence_frame_count: int = 0
        self._speech_frame_count: int = 0

        # Timing
        self.last_activity_time = asyncio.get_event_loop().time()
        self.start_time = asyncio.get_event_loop().time()
        self.is_idle_tool_call: bool = False

        # TTS voice (resolved from profile or config)
        self._tts_voice: str = config.TTS_VOICE

        # Lifecycle flags
        self._shutdown_requested: bool = False
        self._connected_event: asyncio.Event = asyncio.Event()

        # Debouncing for partial transcripts
        self.partial_transcript_task: asyncio.Task[None] | None = None
        self.partial_transcript_sequence: int = 0
        self.partial_debounce_delay = 0.5

    def copy(self) -> "OllamaHandler":
        """Create a copy of the handler."""
        return OllamaHandler(self.deps, self.gradio_mode, self.instance_path)

    # ------------------------------------------------------------------ #
    # Startup & lifecycle
    # ------------------------------------------------------------------ #

    async def start_up(self) -> None:
        """Initialize STT, LLM client, and keep running until shutdown."""
        # 1. Initialize Ollama client
        self.ollama_client = OllamaAsyncClient(host=config.OLLAMA_BASE_URL)

        # 2. Verify Ollama connectivity
        try:
            await self.ollama_client.list()
            logger.info("Connected to Ollama at %s", config.OLLAMA_BASE_URL)
        except Exception as e:
            logger.error("Cannot reach Ollama at %s: %s", config.OLLAMA_BASE_URL, e)
            logger.warning("Proceeding anyway; requests will fail until Ollama is available.")

        # 3. Initialize faster-whisper STT
        try:
            from faster_whisper import WhisperModel

            self.whisper_model = WhisperModel(
                config.STT_MODEL,
                device="auto",
                compute_type="int8",
            )
            logger.info("Loaded faster-whisper model: %s", config.STT_MODEL)
        except Exception as e:
            logger.error("Failed to load STT model '%s': %s", config.STT_MODEL, e)
            logger.warning("Speech-to-text will be unavailable.")

        # 4. Set up conversation with system prompt
        instructions = get_session_instructions()
        self._messages = [{"role": "system", "content": instructions}]
        self._tts_voice = config.TTS_VOICE

        self._connected_event.set()
        logger.info(
            "OllamaHandler ready — model=%s  stt=%s  tts_voice=%s",
            config.MODEL_NAME,
            config.STT_MODEL,
            self._tts_voice,
        )

        # Keep the handler alive until shutdown is requested
        while not self._shutdown_requested:
            await asyncio.sleep(0.1)

    # ------------------------------------------------------------------ #
    # Personality / session management
    # ------------------------------------------------------------------ #

    async def apply_personality(self, profile: str | None) -> str:
        """Apply a new personality (profile) at runtime.

        Updates the system prompt and resets conversation history so the new
        personality takes effect immediately.
        """
        try:
            from reachy_mini_conversation_app.config import config as _config
            from reachy_mini_conversation_app.config import set_custom_profile

            set_custom_profile(profile)
            logger.info(
                "Set custom profile to %r (config=%r)",
                profile,
                getattr(_config, "REACHY_MINI_CUSTOM_PROFILE", None),
            )

            try:
                instructions = get_session_instructions()
            except BaseException as e:
                logger.error("Failed to resolve personality content: %s", e)
                return f"Failed to apply personality: {e}"

            # Reset conversation with new system prompt
            self._messages = [{"role": "system", "content": instructions}]
            logger.info("Applied personality: %s", profile or "built-in default")
            return "Applied personality. Active on next message."
        except Exception as e:
            logger.error("Error applying personality '%s': %s", profile, e)
            return f"Failed to apply personality: {e}"

    async def _restart_session(self) -> None:
        """Reset conversation history (equivalent of restarting a session)."""
        try:
            instructions = get_session_instructions()
            self._messages = [{"role": "system", "content": instructions}]
            logger.info("Session reset (conversation history cleared).")
        except Exception as e:
            logger.warning("_restart_session failed: %s", e)

    # ------------------------------------------------------------------ #
    # Audio receive (microphone) → VAD → STT → LLM → TTS → emit
    # ------------------------------------------------------------------ #

    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        """Receive audio frame from the microphone and run VAD.

        When the user finishes speaking (silence detected), kicks off the
        speech-processing pipeline in a background task.
        """
        if self._shutdown_requested or self.whisper_model is None:
            return

        input_sample_rate, audio_frame = frame

        # Reshape to 1-D mono
        if audio_frame.ndim == 2:
            if audio_frame.shape[1] > audio_frame.shape[0]:
                audio_frame = audio_frame.T
            if audio_frame.shape[1] > 1:
                audio_frame = audio_frame[:, 0]

        # Resample to handler rate if necessary
        if input_sample_rate != HANDLER_SAMPLE_RATE:
            audio_frame = resample(
                audio_frame, int(len(audio_frame) * HANDLER_SAMPLE_RATE / input_sample_rate)
            )

        audio_frame = audio_to_int16(audio_frame)

        # --- simple energy-based VAD ---
        rms = float(np.sqrt(np.mean(audio_frame.astype(np.float32) ** 2)))
        frame_duration = len(audio_frame) / HANDLER_SAMPLE_RATE

        if rms > SILENCE_RMS_THRESHOLD:
            # Voice activity detected
            if not self._is_speaking:
                self._is_speaking = True
                self._speech_frame_count = 0
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
                self._audio_buffer.append(audio_frame)  # keep trailing silence

                silence_duration = self._silence_frame_count * frame_duration
                if silence_duration >= SILENCE_DURATION_S:
                    speech_duration = self._speech_frame_count * frame_duration
                    self.deps.movement_manager.set_listening(False)

                    if speech_duration >= MIN_SPEECH_DURATION_S:
                        logger.debug("Speech ended (%.1fs)", speech_duration)
                        full_audio = np.concatenate(self._audio_buffer)
                        self._audio_buffer = []
                        self._is_speaking = False
                        self._silence_frame_count = 0
                        self._speech_frame_count = 0
                        asyncio.create_task(self._process_speech(full_audio))
                    else:
                        # Too short, discard
                        self._audio_buffer = []
                        self._is_speaking = False
                        self._silence_frame_count = 0
                        self._speech_frame_count = 0

    # ------------------------------------------------------------------ #
    # Speech processing pipeline
    # ------------------------------------------------------------------ #

    async def _process_speech(self, audio_data: NDArray[np.int16]) -> None:
        """Full pipeline: STT → LLM (with tools) → TTS."""
        try:
            # --- 1. Speech-to-text ---
            text = await self._transcribe(audio_data)
            if not text:
                return

            logger.info("User: %s", text)
            await self.output_queue.put(AdditionalOutputs({"role": "user", "content": text}))

            # --- 2. LLM response ---
            self._messages.append({"role": "user", "content": text})
            response_text = await self._chat_with_tools()

            if response_text:
                logger.info("Assistant: %s", response_text)
                await self.output_queue.put(
                    AdditionalOutputs({"role": "assistant", "content": response_text})
                )

                # --- 3. Text-to-speech ---
                await self._synthesize_speech(response_text)

        except Exception as e:
            logger.error("Speech processing error: %s", e)
            await self.output_queue.put(
                AdditionalOutputs({"role": "assistant", "content": f"[error] {e}"})
            )

    async def _transcribe(self, audio_data: NDArray[np.int16]) -> str:
        """Run faster-whisper STT on raw PCM audio."""
        # Resample from handler rate to Whisper's 16 kHz
        float_audio = audio_data.astype(np.float32) / 32768.0
        whisper_audio = resample(
            float_audio,
            int(len(float_audio) * WHISPER_SAMPLE_RATE / HANDLER_SAMPLE_RATE),
        ).astype(np.float32)

        loop = asyncio.get_event_loop()
        segments, _info = await loop.run_in_executor(
            None,
            lambda: self.whisper_model.transcribe(whisper_audio, beam_size=5),
        )

        # Collect all text from segments (run_in_executor returns generator lazily)
        text_parts: list[str] = []
        for seg in segments:
            text_parts.append(seg.text)
        return " ".join(text_parts).strip()

    async def _chat_with_tools(self) -> str:
        """Send conversation to Ollama with tool support; handle tool calls."""
        if self.ollama_client is None:
            return "Ollama client not initialized."

        ollama_tools = self._build_ollama_tools()

        response = await self.ollama_client.chat(
            model=config.MODEL_NAME,
            messages=self._messages,
            tools=ollama_tools or None,
        )

        assistant_msg = response["message"]

        # Handle tool calls if present
        tool_calls = assistant_msg.get("tool_calls")
        if tool_calls:
            # Add the assistant's tool-call message to history
            self._messages.append(assistant_msg)

            for tc in tool_calls:
                func = tc.get("function", {})
                tool_name = func.get("name", "unknown")
                tool_args_dict = func.get("arguments", {})
                tool_args_json = json.dumps(tool_args_dict) if isinstance(tool_args_dict, dict) else str(tool_args_dict)

                try:
                    tool_result = await dispatch_tool_call(tool_name, tool_args_json, self.deps)
                    logger.debug("Tool '%s' result: %s", tool_name, tool_result)
                except Exception as e:
                    tool_result = {"error": str(e)}

                await self.output_queue.put(
                    AdditionalOutputs(
                        {
                            "role": "assistant",
                            "content": json.dumps(tool_result),
                            "metadata": {"title": f"🛠️ Used tool {tool_name}", "status": "done"},
                        }
                    )
                )

                # Handle camera tool image → show in chat
                if tool_name == "camera" and "b64_im" in tool_result:
                    if self.deps.camera_worker is not None:
                        np_img = self.deps.camera_worker.get_latest_frame()
                        if np_img is not None:
                            rgb_frame = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
                        else:
                            rgb_frame = None
                        img = gr.Image(value=rgb_frame)
                        await self.output_queue.put(
                            AdditionalOutputs({"role": "assistant", "content": img})
                        )

                # Add tool result to conversation
                self._messages.append(
                    {
                        "role": "tool",
                        "content": json.dumps(tool_result),
                    }
                )

            # If this was an idle tool call, skip spoken response
            if self.is_idle_tool_call:
                self.is_idle_tool_call = False
                return ""

            # Get follow-up response after tool calls
            follow_up = await self.ollama_client.chat(
                model=config.MODEL_NAME,
                messages=self._messages,
            )
            assistant_msg = follow_up["message"]

        # Extract final response text
        response_text = assistant_msg.get("content", "")
        if response_text:
            self._messages.append({"role": "assistant", "content": response_text})
        return response_text

    @staticmethod
    def _build_ollama_tools() -> list[dict[str, Any]]:
        """Convert internal tool specs to Ollama's expected format."""
        specs = get_tool_specs()
        tools: list[dict[str, Any]] = []
        for spec in specs:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": spec["name"],
                        "description": spec["description"],
                        "parameters": spec["parameters"],
                    },
                }
            )
        return tools

    # ------------------------------------------------------------------ #
    # Text-to-speech
    # ------------------------------------------------------------------ #

    async def _synthesize_speech(self, text: str) -> None:
        """Convert text to speech via pyttsx3 and queue the audio output."""
        if not text.strip():
            return
        
        try:
            import tempfile
            import os

            # Create a temporary file for the WAV output
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_path = tmp_file.name

            try:
                # pyttsx3 is not inherently async, so run in executor
                def _save_to_file():
                    engine = pyttsx3.init()
                    # Set voice if needed
                    voices = engine.getProperty('voices')
                    # Try to find a voice that matches self._tts_voice or just use default
                    for v in voices:
                        if self._tts_voice in v.name or self._tts_voice in v.id:
                            engine.setProperty('voice', v.id)
                            break
                    
                    engine.save_to_file(text, tmp_path)
                    engine.runAndWait()

                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, _save_to_file)

                # Read and decode the WAV file
                if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                    decoded = miniaudio.decode_file(
                        tmp_path,
                        output_format=miniaudio.SampleFormat.SIGNED16,
                        nchannels=1,
                        sample_rate=HANDLER_SAMPLE_RATE,
                    )
                    samples = np.frombuffer(decoded.samples, dtype=np.int16)

                    # Stream audio in ~100 ms chunks
                    chunk_size = HANDLER_SAMPLE_RATE // 10
                    for i in range(0, len(samples), chunk_size):
                        audio_chunk = samples[i : i + chunk_size]
                        if self.deps.head_wobbler is not None:
                            self.deps.head_wobbler.feed(base64.b64encode(audio_chunk.tobytes()).decode("utf-8"))
                        self.last_activity_time = asyncio.get_event_loop().time()
                        await self.output_queue.put(
                            (HANDLER_SAMPLE_RATE, audio_chunk.reshape(1, -1))
                        )
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        except Exception as e:
            logger.error("TTS synthesis error: %s", e)

    # ------------------------------------------------------------------ #
    # Emit (speaker output)
    # ------------------------------------------------------------------ #

    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Emit audio frame to the speaker."""
        # Handle idle
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
        """Send an idle prompt to the LLM to trigger tool-based behaviour."""
        logger.debug("Sending idle signal")
        self.is_idle_tool_call = True
        timestamp_msg = (
            f"[Idle time update: {self.format_timestamp()} - No activity for {idle_duration:.1f}s] "
            "You've been idle for a while. Feel free to get creative - dance, show an emotion, "
            "look around, do nothing, or just be yourself!"
        )
        self._messages.append({"role": "user", "content": timestamp_msg})

        response_text = await self._chat_with_tools()
        if response_text and not self.is_idle_tool_call:
            # Tool handler already reset the flag; speak the response
            await self._synthesize_speech(response_text)

    # ------------------------------------------------------------------ #
    # Voices
    # ------------------------------------------------------------------ #

    async def get_available_voices(self) -> list[str]:
        """Return available pyttsx3 voices."""
        try:
            def _get_voices():
                engine = pyttsx3.init()
                return [v.name for v in engine.getProperty('voices')]
            
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, _get_voices)
        except Exception:
            return ["Default"]

    # ------------------------------------------------------------------ #
    # Shutdown
    # ------------------------------------------------------------------ #

    async def shutdown(self) -> None:
        """Shutdown the handler."""
        self._shutdown_requested = True

        # Cancel any pending debounce task
        if self.partial_transcript_task and not self.partial_transcript_task.done():
            self.partial_transcript_task.cancel()
            try:
                await self.partial_transcript_task
            except asyncio.CancelledError:
                pass

        # Clear remaining items in the output queue
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #

    def format_timestamp(self) -> str:
        """Format current timestamp with date, time, and elapsed seconds."""
        loop_time = asyncio.get_event_loop().time()
        elapsed_seconds = loop_time - self.start_time
        dt = datetime.now()
        return f"[{dt.strftime('%Y-%m-%d %H:%M:%S')} | +{elapsed_seconds:.1f}s]"
