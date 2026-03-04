"""Gemini Live API conversation handler for Reachy Mini.

Drop-in replacement for OpenaiRealtimeHandler using Google's Gemini Live API.

Why Gemini Live:
- Free tier: 1,500 requests/day via Google AI Studio API key
- Same architecture as OpenAI Realtime: single WebSocket, native audio in/out
- VAD, tool calling, interruption handling all built in
- No EU data restrictions (uses Google AI Studio, not Vertex AI)
- Sub-400ms latency (comparable to OpenAI Realtime)

Setup:
1. Get a free API key at https://aistudio.google.com/apikey
2. Add to your .env:
       GEMINI_API_KEY=your_key_here
3. Install dependency:
       /venvs/apps_venv/bin/pip install google-genai
"""

import json
import base64
import asyncio
import logging
import random
from typing import Any, Final, Tuple, Literal, Optional
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import gradio as gr
from google import genai
from google.genai import types
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

# Gemini Live API audio format requirements
GEMINI_INPUT_SAMPLE_RATE: Final[int] = 16000   # 16kHz PCM in
GEMINI_OUTPUT_SAMPLE_RATE: Final[int] = 24000  # 24kHz PCM out
HANDLER_OUTPUT_SAMPLE_RATE: Final[int] = 24000

# Default model — fastest, cheapest, free tier available
DEFAULT_MODEL: Final[str] = "gemini-2.0-flash-live-001"

# Voice options for Gemini Live
GEMINI_VOICES: Final[list[str]] = ["Puck", "Charon", "Kore", "Fenrir", "Aoede", "Leda", "Orus", "Zephyr"]
DEFAULT_VOICE: Final[str] = "Puck"


def _get_api_key() -> str:
    """Resolve Gemini API key from config or environment."""
    import os
    return (
        getattr(config, "GEMINI_API_KEY", None)
        or os.environ.get("GEMINI_API_KEY", "")
    )


def _get_voice() -> str:
    """Resolve voice name from config, falling back to default."""
    voice = getattr(config, "TTS_VOICE", None) or DEFAULT_VOICE
    # If someone left an OpenAI voice name, fall back gracefully
    if voice not in GEMINI_VOICES:
        logger.warning("Voice '%s' not in Gemini voices, using '%s'", voice, DEFAULT_VOICE)
        return DEFAULT_VOICE
    return voice


def _build_tool_config() -> list[dict[str, Any]]:
    """Convert internal tool specs to Gemini function declarations."""
    specs = get_tool_specs()
    declarations = []
    for spec in specs:
        declarations.append({
            "name": spec["name"],
            "description": spec.get("description", ""),
            "parameters": spec.get("parameters", {"type": "object", "properties": {}}),
        })
    return declarations


class GeminiLiveHandler(AsyncStreamHandler):
    """Realtime voice handler using Google Gemini Live API.

    Architecture mirrors OpenaiRealtimeHandler exactly — same fastrtc interface,
    same tool dispatch, same output queue — only the upstream API changes.
    """

    def __init__(
        self,
        deps: ToolDependencies,
        gradio_mode: bool = False,
        instance_path: Optional[str] = None,
    ):
        super().__init__(
            expected_layout="mono",
            output_sample_rate=HANDLER_OUTPUT_SAMPLE_RATE,
            input_sample_rate=GEMINI_INPUT_SAMPLE_RATE,
        )

        self.deps = deps
        self.gradio_mode = gradio_mode
        self.instance_path = instance_path

        self.output_queue: asyncio.Queue = asyncio.Queue()
        self._session: Any = None          # google.genai live session
        self._client: Any = None           # google.genai.Client

        # Timing
        self.last_activity_time: float = 0.0
        self.start_time: float = 0.0
        self.is_idle_tool_call: bool = False

        # Lifecycle
        self._shutdown_requested: bool = False
        self._connected_event: asyncio.Event = asyncio.Event()

        # Partial transcript debouncing (mirrors OpenAI handler)
        self.partial_transcript_task: asyncio.Task | None = None
        self.partial_transcript_sequence: int = 0
        self.partial_debounce_delay: float = 0.5

        # Key tracking for Gradio mode
        self._key_source: Literal["env", "textbox"] = "env"
        self._provided_api_key: str | None = None

    def copy(self) -> "GeminiLiveHandler":
        return GeminiLiveHandler(self.deps, self.gradio_mode, self.instance_path)

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    async def start_up(self) -> None:
        """Connect to Gemini Live API and run until shutdown."""
        loop = asyncio.get_event_loop()
        self.last_activity_time = loop.time()
        self.start_time = loop.time()

        api_key = _get_api_key()

        if self.gradio_mode and not api_key:
            await self.wait_for_args()  # type: ignore[no-untyped-call]
            args = list(self.latest_args)
            textbox_key = args[3] if len(args) > 3 and args[3] else None
            if textbox_key:
                api_key = textbox_key
                self._key_source = "textbox"
                self._provided_api_key = textbox_key
            else:
                api_key = _get_api_key()

        if not api_key:
            logger.warning("GEMINI_API_KEY missing — using placeholder (tests/offline)")
            api_key = "DUMMY"

        self._client = genai.Client(api_key=api_key)

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                await self._run_live_session()
                return
            except Exception as e:
                logger.warning("Live session closed unexpectedly (attempt %d/%d): %s", attempt, max_attempts, e)
                if attempt < max_attempts:
                    delay = (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                    logger.info("Retrying in %.1fs...", delay)
                    await asyncio.sleep(delay)
                else:
                    raise
            finally:
                self._session = None
                try:
                    self._connected_event.clear()
                except Exception:
                    pass

    async def _run_live_session(self) -> None:
        """Open and manage a single Gemini Live session."""
        instructions = get_session_instructions()
        voice = _get_voice()
        model = getattr(config, "MODEL_NAME", DEFAULT_MODEL) or DEFAULT_MODEL

        # Build tool declarations for Gemini
        tool_declarations = _build_tool_config()

        session_config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=instructions,
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                )
            ),
            tools=[types.Tool(function_declarations=tool_declarations)] if tool_declarations else [],
        )

        logger.info("Connecting to Gemini Live — model=%s voice=%s", model, voice)

        async with self._client.aio.live.connect(model=model, config=session_config) as session:
            self._session = session
            self._connected_event.set()
            logger.info("Gemini Live session established")

            # Run sender and receiver concurrently
            sender_task = asyncio.create_task(self._audio_sender())
            try:
                await self._event_receiver()
            finally:
                sender_task.cancel()
                try:
                    await sender_task
                except asyncio.CancelledError:
                    pass

    async def _event_receiver(self) -> None:
        """Receive and process events from Gemini Live session."""
        async for response in self._session.receive():
            if self._shutdown_requested:
                break

            # --- Audio output ---
            if response.data:
                # response.data is raw PCM bytes at 24kHz
                audio_array = np.frombuffer(response.data, dtype=np.int16)
                if self.deps.head_wobbler is not None:
                    self.deps.head_wobbler.feed(base64.b64encode(response.data).decode("utf-8"))
                self.last_activity_time = asyncio.get_event_loop().time()
                await self.output_queue.put(
                    (HANDLER_OUTPUT_SAMPLE_RATE, audio_array.reshape(1, -1))
                )

            # --- Text / transcript ---
            if response.text:
                logger.debug("Assistant text: %s", response.text)
                await self.output_queue.put(
                    AdditionalOutputs({"role": "assistant", "content": response.text})
                )

            # --- Tool / function calls ---
            if response.tool_call:
                for fn_call in response.tool_call.function_calls:
                    tool_name = fn_call.name
                    tool_args = dict(fn_call.args) if fn_call.args else {}
                    call_id = fn_call.id

                    logger.debug("Tool call: %s(%s)", tool_name, tool_args)

                    try:
                        tool_result = await dispatch_tool_call(
                            tool_name, json.dumps(tool_args), self.deps
                        )
                        logger.debug("Tool '%s' result: %s", tool_name, tool_result)
                    except Exception as e:
                        logger.error("Tool '%s' failed: %s", tool_name, e)
                        tool_result = {"error": str(e)}

                    # Emit tool result to UI
                    await self.output_queue.put(
                        AdditionalOutputs({
                            "role": "assistant",
                            "content": json.dumps(tool_result),
                            "metadata": {
                                "title": f"🛠️ Used tool {tool_name}",
                                "status": "done",
                            },
                        })
                    )

                    # Camera tool — show image in chat
                    if tool_name == "camera" and "b64_im" in tool_result:
                        if self.deps.camera_worker is not None:
                            np_img = self.deps.camera_worker.get_latest_frame()
                            rgb = (
                                cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
                                if np_img is not None else None
                            )
                            await self.output_queue.put(
                                AdditionalOutputs({
                                    "role": "assistant",
                                    "content": gr.Image(value=rgb),
                                })
                            )

                    # Send tool result back to Gemini
                    await self._session.send(
                        input=types.LiveClientToolResponse(
                            function_responses=[
                                types.FunctionResponse(
                                    id=call_id,
                                    name=tool_name,
                                    response=tool_result,
                                )
                            ]
                        )
                    )

                    if not self.is_idle_tool_call:
                        # Ask Gemini to speak the result
                        await self._session.send(
                            input="Please respond based on the tool result.",
                            end_of_turn=True,
                        )
                    else:
                        self.is_idle_tool_call = False

                    if self.deps.head_wobbler is not None:
                        self.deps.head_wobbler.reset()

            # --- Server turn complete ---
            if response.server_content and response.server_content.turn_complete:
                logger.debug("Gemini turn complete")
                if self.deps.movement_manager is not None:
                    self.deps.movement_manager.set_listening(True)

            # --- Input transcription (user speech) ---
            if (
                response.server_content
                and response.server_content.input_transcription
                and response.server_content.input_transcription.text
            ):
                transcript = response.server_content.input_transcription.text
                logger.debug("User transcript: %s", transcript)
                await self.output_queue.put(
                    AdditionalOutputs({"role": "user", "content": transcript})
                )

    # ------------------------------------------------------------------
    # Audio input (microphone → Gemini)
    # ------------------------------------------------------------------

    # Buffer for accumulating mic audio before sending
    _mic_buffer: bytes = b""
    _mic_buffer_lock: asyncio.Lock | None = None
    _mic_queue: asyncio.Queue | None = None

    async def _audio_sender(self) -> None:
        """Drain mic queue and forward audio to Gemini Live session."""
        if self._mic_queue is None:
            self._mic_queue = asyncio.Queue()
        while not self._shutdown_requested:
            try:
                audio_bytes: bytes = await asyncio.wait_for(self._mic_queue.get(), timeout=0.1)
                if self._session is not None:
                    await self._session.send(
                        input=types.LiveClientRealtimeInput(
                            audio=types.Blob(data=audio_bytes, mime_type="audio/pcm;rate=16000")
                        )
                    )
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                if not self._shutdown_requested:
                    logger.debug("Audio sender error: %s", e)

    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        """Receive microphone audio and queue it for Gemini."""
        if not self._session or self._shutdown_requested:
            return

        input_sample_rate, audio_frame = frame

        # Flatten to 1-D mono
        if audio_frame.ndim == 2:
            if audio_frame.shape[1] > audio_frame.shape[0]:
                audio_frame = audio_frame.T
            if audio_frame.shape[1] > 1:
                audio_frame = audio_frame[:, 0]
        audio_frame = audio_frame.flatten()

        # Resample to Gemini's required 16kHz if needed
        if input_sample_rate != GEMINI_INPUT_SAMPLE_RATE:
            n_out = int(len(audio_frame) * GEMINI_INPUT_SAMPLE_RATE / input_sample_rate)
            audio_frame = resample(audio_frame, n_out)

        audio_frame = audio_to_int16(audio_frame)
        audio_bytes = audio_frame.tobytes()

        if self._mic_queue is None:
            self._mic_queue = asyncio.Queue()

        try:
            self._mic_queue.put_nowait(audio_bytes)
        except asyncio.QueueFull:
            pass  # drop frame — better than blocking

        # Update listening state
        rms = float(np.sqrt(np.mean(audio_frame.astype(np.float32) ** 2)))
        if rms > 300:
            if self.deps.head_wobbler is not None:
                pass  # let Gemini's VAD handle this
            self.deps.movement_manager.set_listening(True)

    # ------------------------------------------------------------------
    # Emit (speaker output)
    # ------------------------------------------------------------------

    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Emit audio or metadata to the fastrtc Stream."""
        idle = asyncio.get_event_loop().time() - self.last_activity_time
        if idle > 15.0 and self.deps.movement_manager.is_idle():
            try:
                await self.send_idle_signal(idle)
            except Exception as e:
                logger.warning("Idle signal skipped: %s", e)
                return None
            self.last_activity_time = asyncio.get_event_loop().time()

        return await wait_for_item(self.output_queue)  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Personality
    # ------------------------------------------------------------------

    async def apply_personality(self, profile: str | None) -> str:
        """Apply a new personality by restarting the session with new instructions."""
        try:
            from reachy_mini_conversation_app.config import set_custom_profile
            set_custom_profile(profile)
            await self._restart_session()
            return "Applied personality and restarted Gemini Live session."
        except Exception as e:
            logger.error("Error applying personality '%s': %s", profile, e)
            return f"Failed to apply personality: {e}"

    async def _restart_session(self) -> None:
        """Close current session and open a fresh one."""
        try:
            if self._session is not None:
                try:
                    await self._session.close()
                except Exception:
                    pass
                self._session = None
            self._connected_event.clear()
            asyncio.create_task(self._run_live_session(), name="gemini-live-restart")
            try:
                await asyncio.wait_for(self._connected_event.wait(), timeout=5.0)
                logger.info("Gemini Live session restarted.")
            except asyncio.TimeoutError:
                logger.warning("Gemini Live restart timed out; continuing in background.")
        except Exception as e:
            logger.warning("_restart_session failed: %s", e)

    # ------------------------------------------------------------------
    # Idle
    # ------------------------------------------------------------------

    async def send_idle_signal(self, idle_duration: float) -> None:
        """Send idle prompt to Gemini to trigger creative tool-based behaviour."""
        if not self._session:
            return
        logger.debug("Sending idle signal (%.1fs)", idle_duration)
        self.is_idle_tool_call = True
        idle_msg = (
            f"[Idle: {self.format_timestamp()} - {idle_duration:.1f}s idle] "
            "You've been idle. Feel free to get creative — dance, show an emotion, "
            "look around, do nothing, or just be yourself!"
        )
        await self._session.send(input=idle_msg, end_of_turn=True)

    # ------------------------------------------------------------------
    # Voices
    # ------------------------------------------------------------------

    async def get_available_voices(self) -> list[str]:
        """Return available Gemini Live voice names."""
        return GEMINI_VOICES

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        """Gracefully shut down the handler."""
        self._shutdown_requested = True

        if self.partial_transcript_task and not self.partial_transcript_task.done():
            self.partial_transcript_task.cancel()
            try:
                await self.partial_transcript_task
            except asyncio.CancelledError:
                pass

        if self._session is not None:
            try:
                await self._session.close()
            except Exception as e:
                logger.debug("Session close error: %s", e)
            finally:
                self._session = None

        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        self._connected_event.clear()
        logger.info("GeminiLiveHandler shut down.")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def format_timestamp(self) -> str:
        elapsed = asyncio.get_event_loop().time() - self.start_time
        return f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | +{elapsed:.1f}s]"