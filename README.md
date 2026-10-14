# NOVA

A modular, security-first, voice-capable AI desktop assistant for Windows.

> **Status:** All 11 phases implemented and tested — see [Roadmap](#roadmap)
> for what's fully verified here versus what still needs checking on real
> hardware (a mic, a Windows machine, an internet connection that can reach
> Playwright's CDN, etc).

## Overview

NOVA is designed as a local-first agent that turns natural-language voice
(or text) commands into planned, permission-checked, auditable actions on a
Windows machine: opening applications, managing files, driving a browser,
and running developer tooling. It is built to be understandable by a
final-year CS/IT student while following production engineering practice.

```
USER VOICE → VAD → STT → CONTEXT/MEMORY → AGENT → PLANNING →
TOOL SELECTION → SECURITY GATE → TOOL EXECUTION → OBSERVATION →
RE-PLANNING (if needed) → TASK RESULT → TTS → USER
```

## Architecture

```
app/
├── main.py         # Headless-capable entrypoint; wires everything together
├── config/         # Strongly typed settings (pydantic-settings, .env-driven)
├── core/           # Enums, exception hierarchy, logging, agent, planner,
│                   # executor, proactive automation — no I/O, no providers
├── ai/             # LLMProvider protocol + AnthropicProvider (swappable)
├── voice/          # STT/TTS/WakeWord providers, audio capture, pipeline
├── tools/          # Tool base class + Tool Registry
│   ├── windows/    # Process/system/media tools
│   ├── filesystem/ # File read/write/search tools
│   ├── browser/    # Playwright-backed browser session + tools
│   ├── developer/  # git/maven/gradle/npm/docker/run_command tools
│   └── vision/     # Screen-capture + vision-LLM description tool
├── security/       # PermissionManager, ConfirmationManager, CommandPolicy,
│                   # AuditLogger, composed as SecurityGate
├── memory/         # Short-term / long-term / episodic memory
├── database/       # SQLAlchemy async models + engine/session factory
├── events/         # In-process async EventBus decoupling UI from the agent
├── scheduler/      # Lightweight asyncio scheduler for recurring/one-shot jobs
├── ui/             # PySide6 desktop GUI — optional, never required
└── api/            # Optional local API surface
```

**Design rules this project holds itself to:**

- The LLM never touches the OS directly. It can only request a named tool
  with a JSON payload; the Tool Registry validates, permission-checks, and
  executes it.
- LLM provider, voice provider, and database are all behind interfaces
  (`app/ai/base.py`, `app/voice/base.py`, SQLAlchemy) so any one of them can
  be swapped without touching the agent or planner.
- The GUI is optional. Everything in `app/` other than `app/ui` must work
  from a terminal.
- HIGH and CRITICAL risk tool calls require an explicit user confirmation
  before they execute — this is enforced in the registry, not left to
  individual tools to remember.

## Tech Stack

Python 3.12+, `uv`, `asyncio`, Pydantic v2, SQLAlchemy (async, SQLite),
`structlog`, `pytest` + `pytest-asyncio`, `ruff`, `mypy`, PySide6 (GUI,
Phase 9), Playwright (browser, Phase 4), a Whisper-compatible STT
architecture and a pluggable TTS provider (Phase 1 / Phase 10).

Heavier dependencies (`PySide6`, `playwright`, `openai-whisper`,
`sounddevice`) are declared as optional extras (`gui`, `browser`, `voice`)
so Phase 0 setup stays fast and doesn't require a display, microphone, or
downloaded browser binaries. Install them when the phase that needs them
arrives.

## Setup

```bash
cd nova-assistant
# Everything, in ONE command (see the warning below):
uv sync --extra dev --extra voice --extra gui --extra browser
uv run playwright install chromium   # once, for play_youtube_video / browser tools
cp .env.example .env                 # fill in NOVA_LLM_API_KEY etc.
```

> **`uv sync` makes the environment match *exactly* what you ask for — and
> removes everything else.** Run it with only some of your extras and it will
> uninstall the others (e.g. leaving off `--extra browser` removes
> Playwright, which then makes `play_youtube_video` fail with "Browser
> automation isn't set up yet"). It also removes anything you added with
> `uv pip install`, such as `openwakeword` — reinstall it with
> `uv pip install openwakeword` after every `uv sync` if you use `--wakeword`.
> Always pass every extra you want in a single `uv sync` command.

## Running

```bash
uv run python -m app.main              # one-shot text sanity check via the configured LLM
uv run python -m app.main --voice      # push-to-talk voice loop (hold a key)
uv run python -m app.main --wakeword   # always-listening voice loop (say the wake word)
uv run python -m app.main --gui        # push-to-talk voice loop with a PySide6 window
```

`--voice` builds the LLM provider, a default tool registry (system info,
open/close app, file operations, web/YouTube, memory, vision, reminders —
see `app/tools/defaults.py`), a `ToolCallingAgent`, STT/TTS, a microphone
recorder, and a global push-to-talk hotkey listener (`NOVA_PUSH_TO_TALK_KEY`,
default `ctrl+space`), then runs until Ctrl+C. Hold the hotkey, speak,
release — NOVA transcribes, decides whether to call a tool, executes it if
so, and replies out loud. This needs real hardware (mic, speakers) and — on
Linux — a live X/Wayland session for the global hotkey (`pynput`); none of
that exists in this project's own sandbox, so this path is verified
manually rather than in the test suite.

`--wakeword` is the same stack, minus the hotkey: NOVA listens continuously,
gates audio through a lightweight energy-based VAD, and only wakes on the
configured wake word (`app.voice.wakeword.WakeWordEngine`, via
`openwakeword`). **`openwakeword` ships a fixed set of pretrained phrases —
`alexa`, `hey_mycroft`, `hey_jarvis`, `hey_rhasspy` — there is no "nova"
model**, and no way to make it respond to an arbitrary phrase without
training a custom one (a separate, much more involved process: openwakeword's
own training notebook, ~100k synthetic TTS clips). Set `NOVA_WAKE_WORD_MODEL`
to one of the four (default `hey_jarvis`) and say *that* phrase, not "Hey
NOVA". Once triggered, it records for a **fixed 4-second window** to capture
the command rather than detecting when you stop talking — a real, documented
simplification (see `BackgroundListener`'s docstring in
`app/voice/wakeword.py`), not true silence-based endpointing. `openwakeword`
is intentionally not a declared project extra (see the comment in
`pyproject.toml` for why); install it directly with `uv pip install
openwakeword` on Windows/macOS.

`--gui` (needs the `gui` extra: `uv sync --extra gui`, which now also
installs `qasync`) is the same push-to-talk stack as `--voice`, but with a
`MainWindow` (Phase 9) instead of terminal prints: live status
(Listening/Thinking/Speaking), a running transcript and conversation view,
a tool-activity feed, and GUI confirmation dialogs for risky actions
(delete_file, close_application) instead of a console y/n prompt. PySide6's
event loop and asyncio's event loop are two separate things that both want
to own the thread — this bridges them with `qasync` rather than the plain
`asyncio.run()` the other modes use. `VoicePipeline` and `ToolCallingAgent`
both gained an optional `events: EventBus` parameter for this — when given,
they publish VOICE_STARTED/VOICE_STOPPED/TRANSCRIPT_CREATED (pipeline) and
TOOL_STARTED/TOOL_COMPLETED/TOOL_FAILED/RESPONSE_READY (agent), which
`MainWindow.wire_events()` already knows how to render; the console modes
simply don't pass an `EventBus`, so none of this publishing happens there.
Confirmed working up through window construction, full stack wiring
(LLM/memory/registry/agent), and clean shutdown in this project's own
sandbox (via Qt's offscreen platform) — the actual mic/speaker/hotkey path
needs a real desktop, same limitation as `--voice`.

The window uses a dark sci-fi HUD theme (deep-space background, cyan/violet
neon accents, glowing status dot that changes color per state: cyan
listening, amber thinking, violet speaking, red on errors). It's all a QSS
stylesheet plus a few `QGraphicsDropShadowEffect` glows in
`app/ui/main_window.py` — cosmetic only; the update methods and event
wiring underneath are unchanged.

`MicrophoneRecorder` keeps one audio stream open for the whole session and
just toggles a recording flag on press/release, instead of opening a fresh
stream on every press. Opening a stream has real hardware startup latency,
so per-press opening could silently drop the first fraction of a second of
speech — a plausible cause of intermittent Whisper hallucinations
("Thank you.") on short commands even with a perfectly good microphone.
Only the first press of a session pays that cost now.

## Tool Calling

`ToolCallingAgent` (`app/core/agent.py`) is what makes voice commands like
"open notepad" or "what's my CPU usage" actually *do* something instead of
just being described. Each turn: the agent sends the user's message plus
every registered tool's spec (`ToolRegistry.list_tool_specs()`) to the LLM;
if the LLM requests a tool call, the agent executes it through the registry
(so validation, the security gate, and audit logging all still apply) and
asks the LLM once more for a short natural-language summary. It's bounded
to two tool rounds by default (`max_tool_rounds`) — enough for the model to
self-correct once if a tool call fails validation (e.g. it picked the wrong
tool) — not a full multi-step ReAct loop; a request that genuinely needs an
ordered multi-step plan should go through `app.core.planner`/
`app.core.executor` instead (Phase 5).

Both `AnthropicProvider` and `GroqProvider` translate a shared canonical
tool spec (`{"name", "description", "parameters"}`) into their own native
tool-calling format, so `ToolCallingAgent` and `ToolRegistry` never need to
know which provider is active. The same is true for images: both providers
build their own multipart content-block format from a `ChatMessage`'s
`metadata` (`image_base64`/`media_type`), which is what makes
`describe_screen` work regardless of provider — as long as the configured
*model* itself supports vision input (see Memory/Vision/Reminders below).

## Memory, Vision, and Reminders

Three more tool groups, wired in by `app/main.py`'s `_build_stack()` and
optional in `build_default_registry()` (each needs its own collaborator, so
they're opt-in rather than always registered — see that function's
docstring):

- **Memory** (`app/tools/memory/tools.py`): `remember_fact`, `list_memories`,
  `forget_fact`, backed by the Phase 6 `LongTermMemory` and the same SQLite
  database as everything else (`NOVA_DATABASE_URL`) — facts persist across
  restarts, not just within one run. `ToolCallingAgent` also injects the 30 most
  recent stored facts (one line each, length-capped, labelled as notes rather
  than instructions) into its system prompt fresh on every turn (not just when a
  memory tool is explicitly called), so "I like lo-fi" now genuinely changes
  how NOVA answers "play me something" later, in the same or a future
  session, without the model needing to think to look it up.
- **Vision** (`app/tools/vision/screen.py`): `describe_screen` reuses the
  same LLM/model configured for chat. **This only works if that model
  supports image input.** Most Groq text models (gpt-oss, qwen, etc.) do
  not — check `client.models.list()` for a vision-capable one on your
  account, or set `NOVA_LLM_PROVIDER=anthropic` (Claude models support
  vision). If the model can't handle images, this will surface as a normal
  API error from the provider, not a crash.
- **Reminders** (`app/tools/scheduling/tools.py`): `set_reminder` persists a
  one-shot reminder in SQLite and arms it on the Phase 11 `TaskScheduler`;
  when it fires, `main.py` prints it and speaks it via TTS. Pending reminders
  are restored at startup, including overdue reminders, and are marked fired
  only after delivery succeeds.
- **Conversation history** (`app/memory/conversation.py`): user and assistant
  messages are stored in SQLite. NOVA resumes the most recent conversation on
  startup and loads only the configured rolling history window into the model.
- **Audit trail** (`app/security/manager.py`): permission decisions,
  confirmations, tool completions, failures, and errors are appended to SQLite.
  Common credential fields are redacted before payloads are stored.

## Configuration

All settings are environment variables prefixed `NOVA_`, loaded from the
process environment or a local `.env` file (see `.env.example` for the full
list — LLM provider/model/key, voice provider, database URL, browser
headless mode, security confirmation toggles, and logging).

`app.ai.factory.build_llm_provider` picks the concrete `LLMProvider` from
`NOVA_LLM_PROVIDER`. Two are implemented today:

- `anthropic` (default) — `NOVA_LLM_MODEL=claude-sonnet-4-6`, needs a paid key.
- `groq` — `NOVA_LLM_MODEL=llama-3.3-70b-versatile`, has a free tier (no
  credit card, ~30 req/min / 14,400 req/day at time of writing) — get a key
  at [console.groq.com/keys](https://console.groq.com/keys). Good for
  running NOVA's voice/agent loop without paying for the Anthropic API.

Adding another provider (OpenAI, a local model) means writing one class in
`app/ai/` implementing `LLMProvider` and adding one branch to the factory —
nothing else in the agent, planner, or executor needs to change.

`app.voice.factory.build_stt_provider`/`build_tts_provider` do the same for
voice, driven by `NOVA_VOICE_PROVIDER`:

- `whisper_local` — `faster-whisper` for STT (downloads a model, runs
  offline) + `pyttsx3` for TTS (OS-native voices, fully offline).
- `groq` — uses `NOVA_VOICE_API_KEY`, or reuses `NOVA_LLM_API_KEY` **only when
  `NOVA_LLM_PROVIDER=groq` too** (an Anthropic key is never sent to Groq), for both `whisper-large-v3-turbo`
  transcription and `canopylabs/orpheus-v1-english` speech, so one free key
  drives the whole voice loop with no local model download. Chat and STT
  have generous free-tier quotas; **TTS's free tier is much tighter**
  (roughly 200 characters/request, low daily cap) — fine for short spoken
  replies, not for reading long text aloud. Check current limits at
  [console.groq.com/settings/billing](https://console.groq.com/settings/billing).
- `none` (default) — `NullSTTProvider`/`NullTTSProvider`, no audio at all.

## Testing

```bash
uv run pytest                 # run the test suite (359 tests)
uv run pytest --cov=app --cov-report=term-missing   # with coverage (87% overall)
uv run ruff check .           # lint
uv run mypy app               # type-check (strict mode)
```

Unit tests require no real API calls, microphone, browser, or network
access — everything is exercised in isolation or with mocks/fakes. GUI tests
run against Qt's offscreen platform plugin (`tests/conftest.py` sets
`QT_QPA_PLATFORM=offscreen` automatically).

## Security Model

- **Risk levels** (`app.core.enums.RiskLevel`): `LOW`, `MEDIUM`, `HIGH`,
  `CRITICAL`. Every tool declares one.
- **LOW** — read-only, non-sensitive (system info, open an app, search files).
- **MEDIUM** — creates or modifies non-critical state (create/modify a file,
  download something).
- **HIGH** — destructive or system-altering (delete a file, run a shell
  command, install software). Requires confirmation.
- **CRITICAL** — irreversible or credential-related operations. Requires
  confirmation and is logged to the audit trail regardless of outcome.
- The security layer (`app/security`, built out in Phase 7) is the single
  gate every tool call passes through — no tool decides for itself whether
  it's allowed to run.

## Tool Architecture

Tools are declared once (`app.tools.base.Tool` + `ToolMetadata`) with a
name, description, JSON input/output schema, and risk level, then
registered with a central Tool Registry (Phase 2). The agent selects tools
by name from that registry — it never imports a tool implementation
directly.

## Roadmap

All 11 phases below are implemented and tested (359 tests passing, `ruff check` and
`mypy --strict` clean). A few integration points genuinely need hardware or
resources this build environment doesn't have (a real microphone/speaker, a
downloaded Chromium binary, a Windows machine, an installable wake-word model
for Python 3.12) — those are called out under **Known limitations** below
rather than glossed over.

| Phase | Focus | Status |
|---|---|---|
| 0 | Foundation: config, logging, core interfaces, tests | ✅ |
| 1 | Voice conversation: mic → STT → LLM → TTS | ✅ (see limitations) |
| 2 | Tool calling: Tool interface, Tool Registry, basic Windows tools | ✅ |
| 3 | File operations | ✅ |
| 4 | Browser automation (Playwright) | ✅ (see limitations) |
| 5 | Multi-step planning and re-planning | ✅ |
| 6 | Memory (conversation, preferences, task history) | ✅ — wired into the voice loop |
| 7 | Security (permissions, confirmations, audit log) | ✅ |
| 8 | Developer tools (git, maven, gradle, npm, docker) | ✅ (see limitations) |
| 9 | PySide6 GUI | ✅ — wired into `--gui` (see limitations) |
| 10 | Wake word + background listening | ✅ — wired into `--wakeword` (see limitations) |
| 11 | Advanced agent: vision, screen understanding, proactive automation, scheduled tasks | ✅ — vision + reminders wired into the voice loop |

### Known limitations (honest accounting)

These are real gaps, not implementation shortcuts hidden behind passing tests:

- **Voice (Phase 1) + live Groq voice:** `WhisperSTTProvider`/`Pyttsx3TTSProvider`
  (local) and `GroqSTTProvider`/`GroqTTSProvider` (hosted, free-tier-friendly)
  are real, complete implementations, unit-tested with mocked/fake audio —
  this sandbox has no microphone or speaker. They have been confirmed
  working end-to-end on a real Windows machine with a live Groq key,
  including actual audio playback.
- **Push-to-talk hotkey (`app/voice/hotkey.py`):** the combo-matching logic
  is pure and fully unit-tested, but `pynput` needs a live display/input
  backend just to *import* — this sandbox has neither, so `PushToTalkListener.start()`
  has only been exercised via its real (not mocked) `ImportError → VoiceError`
  failure path here. `run_voice_loop()` (`python -m app.main --voice`) needs
  verification on a real desktop with a mic and speakers.
- **Browser (Phase 4):** `BrowserSession`/the browser tools are real Playwright
  code, but this environment can't download a Chromium binary (the sandbox's
  network allowlist doesn't include Playwright's CDN). Tests exercise the
  tools against a fake Playwright-shaped page. Run `uv sync --extra browser &&
  playwright install chromium` on a machine with normal internet access to use
  them for real — this also applies to `play_youtube_video`
  (`app/tools/web/youtube.py`), which needs a real Chromium install to
  actually click a YouTube result and its native "Skip Ad" button; `open_webpage`
  (stdlib `webbrowser`) needs no extra install and always works, but only
  opens a page — it can't click or interact with it, since it launches the
  user's own browser as an untracked OS process rather than one Playwright
  controls. Searching and clicking the first result has been confirmed
  working manually on a real Windows machine; the ad-skip background loop
  is unit-tested with a fake page but not yet confirmed against a real ad.
- **Windows-native tools (Phase 2):** `open_application`/`close_application`
  work cross-platform; `set_volume` (pycaw) and parts of `take_screenshot`
  (Pillow's `ImageGrab`) are Windows/display-specific and were only
  exercisable here via their graceful-failure paths (no COM, no display).
  They should be re-verified on an actual Windows machine.
- **Developer tools (Phase 8):** `git_status`/`git_log`/`run_python`/`npm`
  were tested against real binaries in this sandbox. `mvn`, `gradle`, and
  `docker` aren't installed here, so those three were only verified via their
  "binary not found" failure path — the happy path should be re-checked
  wherever those tools actually exist.
- **GUI (Phase 9) + `--gui` mode:** `MainWindow` is real PySide6 code, tested
  end-to-end (widget updates, event-bus wiring, confirmation dialog logic)
  using Qt's offscreen platform plugin, since this sandbox has no display
  server. `run_gui_voice_loop`'s full wiring (qasync event loop, LLM,
  memory, registry, agent, window construction, clean shutdown) has been
  smoke-tested successfully in this sandbox up through the point where it
  needs a real microphone — the actual visual appearance, live status
  updates during a real conversation, and GUI confirmation dialogs haven't
  been seen on an actual desktop yet. One real bug was found and fixed here:
  on a mic or hotkey setup failure, the coroutine used to return immediately,
  which ends `loop.run_until_complete()` and closes the whole window before
  Qt ever paints the error message onto the Logs tab — the window would just
  flash and vanish. It now stays open (`_stay_open_after_setup_failure`,
  regression-tested) so the error is actually visible.
- **Wake word (Phase 10):** `EnergyVAD`, `BackgroundListener`, and the
  `WakeWordEngine` wrapper are implemented and unit-tested (with a fake wake
  detector and a fake chunk source), including the fixed-window command
  recording after trigger, a hook that drains the monitoring queue's backlog
  after each triggered command (so monitoring resumes at the present moment
  instead of working through stale audio), and resilience to a single
  failed turn (a transient provider error no longer crashes the whole
  always-listening loop — confirmed by test, this was a real bug before the
  fix). `openwakeword` genuinely installs fine on Windows/macOS (its
  `tflite-runtime` dependency is Linux-only in its own metadata; only
  `onnxruntime`, which has current wheels, is needed there) — but this
  sandbox is Linux, so the real model has never actually been loaded or
  tested against a live "Hey NOVA" here. `ContinuousMicrophoneChunkSource`'s
  queue/draining logic is unit-tested with a fake audio backend; its actual
  hardware behavior, and the dual-stream design `--wakeword` uses (see
  `run_wakeword_loop`'s docstring), are untested on real hardware.
- **Tool calling with a real LLM:** confirmed working end-to-end on a real
  Windows machine — multi-tool turns (opening Chrome + a webpage in one
  reply), self-correction when a tool call fails validation, and non-English
  speech (Hindi/Urdu) all worked in practice, not just in unit tests.
  `describe_screen`, `remember_fact`/`list_memories`, and `set_reminder`
  are unit-tested but not yet confirmed live — in particular, `describe_screen`
  needs a vision-capable model, and neither this project's Groq testing nor
  this sandbox has one confirmed available; check `client.models.list()`
  or use Anthropic if it errors.

## Screenshots

_No visual walkthrough yet — the GUI has only been run headlessly (offscreen)
in this environment. Screenshots to be added once verified on a real desktop._

## Future Improvements

- Swap the in-memory event bus for a persisted one if multi-process ever
  becomes necessary (not planned; deliberately avoided for now).
- Introduce a vector store behind the same `MemoryManager` interface once
  semantic recall is needed (Phase 6 ships with SQLite only).
- Add a local-model `LLMProvider` implementation for fully offline use.


## Safety defaults

- **File tools are confined** to `NOVA_WORKSPACE_ROOT` (default: your home
  folder). Credential files (`.env`, `.ssh/`, `*.pem`, ...) are always refused,
  and the workspace root / home folder / drive roots can never be deleted or moved.
  Overwriting an existing file (write/copy/move) asks for confirmation first.
- **Developer tools** (`run_command`, git, npm, maven, gradle, docker) are off
  unless `NOVA_ENABLE_DEVELOPER_TOOLS=true`; the ones that execute code always ask.
- **Tool output is untrusted**: the model is told never to follow instructions
  found in file contents or web pages, and memory facts are stored as single
  length-capped lines labelled as notes, not instructions.
- **Voice recordings are not kept** unless `NOVA_DEBUG_SAVE_AUDIO=true` (a very
  quiet recording is still saved so the "check your microphone" hint works);
  transcripts are logged by length only.
- `close_application` matches the exact process name (>= 3 characters), never
  touches core system processes, and never kills NOVA or its parent shell.


## Wake word ("Hey NOVA")

`--wakeword` mode uses [openWakeWord](https://github.com/dscripka/openWakeWord),
which ships exactly four ready-made phrases: `alexa`, `hey_mycroft`,
`hey_jarvis` (the default) and `hey_rhasspy`. **There is no ready-made "Hey
NOVA" model**, so out of the box you say one of those phrases:

```
NOVA_WAKE_WORD_MODEL=hey_jarvis
```

To get a real "Hey NOVA", train a custom model with openWakeWord's own training
notebook (see "Training New Models" in its repository; it generates synthetic
speech, so no recording is required) and point NOVA at the file it produces:

```
NOVA_WAKE_WORD_MODEL=C:\models\hey_nova.onnx      # or a .tflite file
```

If the wake word is rarely detected, lower `NOVA_WAKE_VAD_THRESHOLD` (default
200); in a noisy room raise it. Every audio chunk is fed to the model, and the
loudness check only confirms a detection.

## Window accessibility

- **Type instead of speak**: the text box at the top sends a request (Enter);
  tick "Read replies aloud" to have NOVA speak the reply. Works with no
  microphone at all, and with no global hotkey.
- **Keyboard**: Tab order is text box, Send, read-aloud, talk button, tabs.
  Hold Space on the focused talk button to talk. `Ctrl+L` focuses the text box;
  `Ctrl+=` / `Ctrl+-` / `Ctrl+0` make all text larger, smaller, or reset it
  (80%-200%). The focused control has a thick cyan outline.
- **Screen readers**: every control and list has an accessible name and
  description, the status is announced as it changes, and each NOVA reply is
  announced when it arrives.
- **Contrast**: all text colours meet WCAG AA (4.5:1) on every panel.
- The on-screen talk button, which was previously not connected to anything,
  now works. When voice is unavailable the button says so and is visibly
  disabled; the window remains a fully working text assistant.
