# Review fixes

Two passes: the first fixed the bugs found by reading the core; the second ("Pass 2" below)
was a full-codebase audit: every module read, coverage + vulture + bandit, fuzzing of all
tool inputs, and a re-review of the pass-1 patches themselves.

All changes come with regression tests (`tests/unit/test_review_fixes.py`, `test_review_fixes_2.py`, `test_tool_fuzz.py` plus
additions to `test_gui.py`, `test_main_gui_helpers.py`, `test_defaults.py`,
`test_tool_calling_agent.py`). 470+ tests pass; ruff clean.

**After unzipping run `uv lock` (or just `uv run ...`) once: Pillow was added to
`pyproject.toml`, so `uv.lock` is out of date.**

## Bugs fixed
| Area | Fix |
|---|---|
| Agent | Tool `data` now reaches the LLM (truncated to 6000 chars); results go back as a *user* turn, not a trailing assistant message; unexpected tool exceptions no longer crash the turn; turns are serialized with a lock; history never starts on an assistant turn |
| Event bus | A failing subscriber no longer breaks publish / other subscribers |
| Registry / security | Tool-level checks (path confinement) run *before* the confirmation prompt; audit records the decision after the prompt; audit-store failure no longer blocks tools; long payloads are shortened in logs and dialogs |
| File tools | Confined to a workspace root (default: home dir); credential files refused; overwriting via write/copy/move asks for confirmation; `search_files` validates patterns, caps results, runs off the event loop; `create_file` validates content |
| TTS | Playback runs in a worker thread (no more frozen app/GUI); long replies are chunked to Groq's 200-char limit; a TTS failure no longer discards the reply |
| Reminders | A delivery failure marks the reminder `failed` instead of re-firing on every startup; GUI shows the reminder before speaking |
| `get_processes` | Real CPU sampling (was always 0.0) and validated `limit` |
| `describe_screen` | Portable temp file (was hard-coded `/tmp`), cleaned up; Pillow declared |
| Planner / executor | Malformed planner JSON -> `PlanningError`; crashing tools fail their step, not the run |
| Settings | Model default follows provider; the previously-ignored security settings are wired in; `format` denylist entry made specific |
| Voice | Hotkey callback errors no longer kill the listener; wake-word loop survives unexpected errors; VAD threshold configurable (`NOVA_WAKE_VAD_THRESHOLD`) |
| GUI | Closing the window now runs cleanup (previously the loop died mid-coroutine); startup failures are shown in the window; crashed turns reset status to Idle |
| Misc | Ctrl+C exits cleanly; memory ordering + dedupe; finished scheduler jobs removed; Playwright browser closed at shutdown; Windows path handling in `run_command`; `.gitignore` covers `data/*.wav` |

## Behavior changes you should know about
* File tools can no longer touch anything outside your home folder (set `NOVA_WORKSPACE_ROOT` to change).
* `write_file` / `copy_file` / `move_file` ask for confirmation only when the destination already exists.
* Developer tools (`run_command`, git, npm, ...) are opt-in via `NOVA_ENABLE_DEVELOPER_TOOLS=true`.

## Not changed (design limitations)
* Wake word: a phrase split across two 1-second chunks can still be missed.
* `NOVA_BROWSER_HEADLESS` is still unused (no registered tool needs it yet).

## Pass 2 - additional findings and fixes

| Severity | Finding | Fix |
|---|---|---|
| High | `close_application` matched by *substring*: `name="e"` terminated every process containing an "e" (explorer, NOVA itself, your terminal) | Exact-name match, >= 3 chars, core system processes protected, never kills NOVA or its parent shell |
| High | `open_application` handed any string to `os.startfile`, which launches *any file* (`.bat`, `.js`, `.hta`, installers) - LOW risk, no confirmation | Bare program names only; paths, spaces and script/installer extensions rejected |
| High | Code-execution chain: `create_file x.py` then `run_python x.py` - neither asked for confirmation | `run_python`, maven, gradle, npm now always confirm |
| High | `delete_file` / `move_file` accepted the workspace root, so one "yes" could wipe the home folder | Workspace root, home folder and drive roots are protected |
| High | Browser `open_url` accepted `file://`, letting `extract_text` read local secrets past the file confinement | http/https only; screenshots limited to image extensions |
| High | Voice factory sent `NOVA_LLM_API_KEY` to Groq even when the LLM provider was Anthropic (key leak + 401) | New `NOVA_VOICE_API_KEY`; the LLM key is reused only if the LLM is Groq |
| Medium | Prompt injection: tool output (web pages, files) reached the model unlabelled; stored facts went into the *system* prompt verbatim | Results labelled untrusted + system prompt rule; facts are single-line, length-capped notes; model told never to save memories because a tool result said so |
| Medium | Developer tools and `describe_screen` had no parameter schema, so the model never saw their argument names | Schemas added |
| Medium | A failed LLM call (rate limit) left a dangling user message, saved to the DB, so a retry sent the request twice | Turn state is rolled back; the user message is persisted only after the LLM accepts it |
| Medium | Your voice was written to `data/debug_last_recording.wav` on every turn; transcripts were logged in full | Opt-in via `NOVA_DEBUG_SAVE_AUDIO` (quiet recordings still saved for the mic hint); only transcript length is logged |
| Low | Fuzzing found 9 crashes in 5 tools (NUL bytes in paths, `limit=inf`, unhashable `direction`) | Fixed; `test_tool_fuzz.py` sweeps every tool with garbage input |
| Low | `set_volume` broke on current pycaw (`GetSpeakers()` API change); `bool` accepted as a level | Handles both APIs; bool rejected |
| Low | Missing/invalid config printed a raw traceback in console mode | One clear line, exit code 1 |

## Pass 3 - GUI accessibility and the wake word

**Correction to pass 2:** the debug-audio fix only covered the console loop. The GUI
loop still saved every recording; both loops now share one gated helper
(`_handle_captured_audio`) and a test guards against a loop saving audio itself.

| Area | Finding | Fix |
|---|---|---|
| GUI | The on-screen HOLD TO TALK button was never connected to anything | Press/release now drive the same path as the hotkey (mouse and Space key) |
| GUI | No way to type a request; no mic or no hotkey made the window useless | Text box + Send + "Read replies aloud"; missing mic/hotkey are no longer fatal |
| GUI | Status stayed on SPEAKING (after a reply) or THINKING (after a silent press) until the next turn | Every voice turn now ends at IDLE |
| GUI | Two palette colours were 3.8-4.2:1 (below WCAG AA 4.5:1) | Lightened to >= 5.8:1; a test checks every text colour on every panel |
| GUI | No accessible names, no visible keyboard focus, default tab order started at the talk button | Names/descriptions on every control, thick focus ring, logical tab order, Ctrl+L, text size Ctrl+= / Ctrl+- / Ctrl+0 |
| GUI | Disabled voice button looked fully active | Dimmed, no glow, label says why |
| GUI | Fire-and-forget typed-turn tasks could be garbage-collected mid-run | References kept until done |
| Wake word | Quiet chunks were never fed to the model, which keeps rolling audio state, so phrases that started in a "quiet" chunk were cut in half | Every chunk is fed; loudness only validates a hit (current or previous chunk) |
| Wake word | Only 4 pretrained phrases exist - "Hey NOVA" can't work out of the box | `NOVA_WAKE_WORD_MODEL` accepts a path to a custom `.onnx`/`.tflite` model; README explains how to train one |

`tests/unit/test_gui_loop_integration.py` drives the real `run_gui_voice_loop()` headlessly (offscreen Qt +
qasync, with the mic/hotkey/LLM faked) and is what caught the stuck-status bug.
