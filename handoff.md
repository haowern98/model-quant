# Model Surgery Test Recipe Handoff

## Current Branch State

- Current branch: `feat/rebuild-test-recipe-evals`
- Branch base: `c5666a4 feat: add single and compare recipe eval modes`
- This is the point before the first official EleutherAI `lm-eval` backend/API attempt.
- Previous uncommitted work was stashed before branching:
  - `wip before branching from c5666a4`
- Remaining dirty state after branch creation:
  - `native/cpp/llama.cpp` submodule marker was already dirty.

## Mandatory Handoff Maintenance Rule

The next model or developer must keep this file updated while working.

Every meaningful change must be appended to `handoff.md`, including:

- Files added, edited, moved, or deleted.
- New commands, APIs, IPC commands, routes, or native exports.
- Eval/task behavior changes.
- Build system changes.
- Test commands run and whether they passed or failed.
- Known bugs, regressions, limitations, and workarounds.
- Design decisions and why they were made.

Do not leave the next handoff to memory or chat history. If a change affects the project, write it here before ending the turn.

## What `c5666a4` Contains

This commit has the native recipe eval foundation:

- Single recipe eval mode.
- Compare baseline vs recipe eval mode.
- Native llama.cpp runtime path.
- In-memory recipe conversion path from earlier commits.
- `evals/smoke_texts.json`.
- PPL-style smoke evaluation.
- Runtime metrics such as prompt eval, token generation, TTFT, load time, elapsed time, tensor count, VRAM peak, working set, disk size.

This commit does **not** contain:

- `evals/standard_subset.json`.
- Internal `standard_subset` task eval.
- Official EleutherAI `lm-eval` backend install gate.
- Local HTTP eval API adapter.
- `src-tauri/src/commands/eval_api.rs`.

## What Was Tried After `c5666a4`

The later work attempted to add a more serious `Test Recipe` evaluation workflow.

### 1. Internal Standard Eval Subset

Added `evals/standard_subset.json` with 2 hand-written samples each for:

- `arc_challenge`
- `arc_easy`
- `gsm8k_small`
- `hellaswag`
- `mmlu_mixed`
- `truthfulqa_mc`

Problem:

- `n=2` per task was too small to show meaningful quality changes.
- It was useful only as a smoke test, not a benchmark.

### 2. Official EleutherAI lm-eval Backend

Added an official backend installer under `%LOCALAPPDATA%\MSGEval`.

Installed packages:

- `lm-eval==0.4.12`
- `transformers>=4.56,<5`

Generated adapter files:

- `%LOCALAPPDATA%\MSGEval\model_surgery_lm_eval\__init__.py`
- `%LOCALAPPDATA%\MSGEval\model_surgery_lm_eval_runner.py`

The app attempted to run `lm-eval` from Tauri via Python subprocesses.

Initial intent:

- `Official Core Eval`: use official lm-eval tasks.
- `Standard Eval`: later changed to use selected official lm-eval tasks.
- Keep `PPL Smoke` as native quick check.

### 3. Custom lm-eval Model Adapter

Created a custom lm-eval model named:

- `model_surgery_api`

The Python adapter called a local app-owned HTTP API:

- `POST /v1/loglikelihood`
- `POST /v1/loglikelihood_rolling`
- `POST /v1/generate_until`
- `GET /health`

The adapter implemented lm-eval methods:

- `loglikelihood`
- `loglikelihood_rolling`
- `generate_until`

This was a valid lm-eval integration approach, but the backend design was not good enough yet.

### 4. Local Rust HTTP Eval API

Added:

- `src-tauri/src/commands/eval_api.rs`

The local API listened on `127.0.0.1:<ephemeral port>` and used bearer-token auth.

It called native FFI wrappers for:

- baseline loglikelihood
- recipe loglikelihood
- rolling loglikelihood
- generation

Debug logging was added to:

- `%LOCALAPPDATA%\MSGEval\api-<port>.log`

This log was important for diagnosing socket/request issues.

### 5. Native Runtime FFI Additions

Added native C++ exports and Rust bindings for:

- `ms_runtime_loglikelihood_baseline`
- `ms_runtime_loglikelihood_recipe`
- `ms_runtime_rolling_loglikelihood_baseline`
- `ms_runtime_rolling_loglikelihood_recipe`
- `ms_runtime_generate_baseline`
- `ms_runtime_generate_recipe`

Corresponding Rust wrappers were added in:

- `src-tauri/src/ffi/runtime_bindings.rs`

## Bugs Encountered And Fixed During The Attempt

### Python Adapter Constructor Conflict

Error:

```text
TypeError: model_surgery_lm_eval.ModelSurgeryAPI() got multiple values for keyword argument 'batch_size'
```

Cause:

- lm-eval already passes `batch_size`.
- The adapter also used `batch_size` as a constructor arg.

Fix tried:

- Renamed adapter parameter to `api_batch_size`.

### Windows Unicode stdout Crash

Error:

```text
UnicodeEncodeError: 'charmap' codec can't encode character '\u2191'
```

Cause:

- lm-eval prints arrows and symbols.
- Windows Python stdout defaulted to `cp1252`.

Fix tried:

- Set child process env:
  - `PYTHONUTF8=1`
  - `PYTHONIOENCODING=utf-8`
  - `HF_HUB_DISABLE_SYMLINKS_WARNING=1`

### Windows Socket Abort

Error:

```text
urllib.error.URLError: <urlopen error [WinError 10053] An established connection was aborted by the software in your host machine>
```

API log showed:

```text
request read failed: failed to read eval API request: A non-blocking socket operation could not be completed immediately. (os error 10035)
```

Cause:

- The TCP listener was non-blocking.
- Accepted sockets inherited behavior that caused reads to return `WouldBlock` on Windows.
- The API closed the socket before Python finished sending the POST.

Fix tried:

- Set accepted sockets to blocking mode with `stream.set_nonblocking(false)`.

### Misleading Failure Status

Problem:

- Official lm-eval completed, but the modal showed `FAIL`.

Cause:

- The modal treated `tokenGenTps > 0` as pass/fail.
- Official lm-eval result objects did not populate native runtime TPS fields.

Fix tried:

- For official eval modes, show `EVAL OK` if lm-eval process completed successfully.

### Wrong Multiple Choice Metric

Problem:

- ARC showed `0%` even though lm-eval JSON had `acc_norm=1.0`.

Cause:

- Parser preferred `acc,none` before `acc_norm,none`.

Fix tried:

- Prefer `acc_norm,none` over raw `acc,none`.

### Fake Runtime Rows

Problem:

- Official eval modal showed zero prompt eval/token gen/VRAM rows.

Cause:

- Official lm-eval path did not return native benchmark fields.

Fix tried:

- Added a separate official eval runtime summary instead of showing fake native runtime rows.

## What Worked

The official lm-eval integration eventually did run end-to-end for small task sets.

Example successful result shown in the app:

- `EVAL OK`
- Standard Eval task table populated.
- Baseline and recipe scores displayed.
- Example elapsed:
  - Baseline around `228.7s`
  - Recipe around `1531.1s`

The local API log confirmed model requests completed:

```text
POST /v1/loglikelihood ...
/v1/loglikelihood complete
```

The path worked functionally, but the performance was unacceptable.

## What Failed

### Full `mmlu` Was Accidentally Too Large

Standard Eval was changed to use:

```text
arc_challenge,arc_easy,gsm8k,hellaswag,mmlu,truthfulqa_mc2 --limit 10
```

Problem:

- `mmlu` is a group with many subtasks.
- `--limit 10` applies per MMLU subtask.
- This produced thousands of loglikelihood calls, not a small 10-sample benchmark.

Observed live run:

```text
1044 loglikelihood POSTs
1043 completed
still running baseline pass
```

The run appeared to loop for nearly 30 minutes or more because it was still processing thousands of requests.

### Per-Request Model Loading Made It Too Slow

The local API design effectively did this:

```text
request -> load model -> apply recipe/conversion -> score -> unload
```

This caused:

- Sawtooth VRAM pattern.
- Very slow recipe pass.
- Repeated load/conversion cost.
- Long Standard Eval runtime even with small sample counts.

The intended design should be:

```text
baseline pass:
  load baseline once
  score all lm-eval requests
  unload

recipe pass:
  load recipe once
  score all lm-eval requests
  unload
```

### `api_batch_size=1` Was Too Slow

The initial adapter sent one request at a time.

Later changed to:

```text
api_batch_size=8
```

This helped somewhat, but it did not solve the deeper issue because each native API call still loaded the model.

### Official Backend Gives Poor UI Feedback

The app only showed:

```text
Running official lm-eval baseline pass...
```

For a long time, with no detail.

lm-eval can spend time:

- starting Python
- importing packages
- loading task YAMLs
- downloading/checking datasets
- building contexts
- running many loglikelihood requests

The app did not surface those phases.

## Validation Commands Used

Task validation:

```powershell
& "$env:LOCALAPPDATA\MSGEval\.venv\Scripts\python.exe" -m lm_eval validate --tasks arc_challenge,arc_easy,gsm8k,hellaswag,mmlu,truthfulqa_mc2
```

Later smaller MMLU-subject validation:

```powershell
& "$env:LOCALAPPDATA\MSGEval\.venv\Scripts\python.exe" -m lm_eval validate --tasks arc_challenge,arc_easy,gsm8k,hellaswag,mmlu_high_school_physics,mmlu_college_computer_science,mmlu_professional_medicine,truthfulqa_mc2
```

Both validated successfully.

Build checks repeatedly used:

```powershell
npm run build
cargo check
cargo build --release
```

Release exe path:

```text
C:\Users\Wu Family Computer\Downloads\Project 2\src-tauri\target\release\model-surgery.exe
```

## Important Logs And Paths

Official eval backend:

```text
C:\Users\Wu Family Computer\AppData\Local\MSGEval
```

lm-eval run logs:

```text
C:\Users\Wu Family Computer\AppData\Local\MSGEval\runs\<baseline|recipe>-<timestamp>\lm-eval.log
```

Local API logs:

```text
C:\Users\Wu Family Computer\AppData\Local\MSGEval\api-<port>.log
```

These API logs show whether lm-eval reached the app and how many loglikelihood requests completed.

## Recommended Restart Plan

Do **not** immediately rebuild the official backend path the same way.

Recommended sequence:

1. Start from `c5666a4`.
2. Keep native single/compare eval modes.
3. Add a small internal Standard Eval first, but generate or import enough real samples.
4. If using official lm-eval again, implement model/session reuse first.
5. Only after model/session reuse works, add official lm-eval task integration.
6. Add clear cancellation before long-running evals.
7. Add live progress states before enabling heavy official evals.

## Better Architecture For Next Attempt

### Minimum Acceptable Official Eval Architecture

The local API should support sessions:

```text
POST /v1/session/start
  mode: baseline | recipe
  model_path
  recipe_targets

POST /v1/loglikelihood
  session_id
  requests[]

POST /v1/generate_until
  session_id
  requests[]

POST /v1/session/end
  session_id
```

Native runtime should keep one model loaded per session.

For compare mode:

```text
start baseline session
run baseline lm-eval
end baseline session

start recipe session
run recipe lm-eval
end recipe session
```

Never keep baseline and recipe loaded at the same time.

### Safer Standard Eval Task Set

Avoid full `mmlu`.

Use selected subjects:

```text
arc_challenge
arc_easy
gsm8k
hellaswag
mmlu_high_school_physics
mmlu_college_computer_science
mmlu_professional_medicine
truthfulqa_mc2
```

For a first useful default:

```text
--limit 10
```

That gives a reasonable smoke-quality set without exploding into all MMLU subjects.

### UI Requirements For Next Attempt

Add progress messages such as:

- Installing official eval backend.
- Starting Python.
- Loading task configs.
- Preparing datasets.
- Starting baseline model session.
- Running baseline task requests.
- Baseline complete.
- Starting recipe model session.
- Running recipe task requests.
- Recipe complete.

Add a cancel button before any official eval mode is exposed.

## User Preferences Captured

- User wants quality-vs-VRAM testing.
- User does not want temporary GGUF writes for testing.
- User accepts DLL/native runtime if the app still feels like one app.
- User wants model-agnostic support, not Gemma-only.
- User wants tokenizer/runtime behavior to use llama.cpp, not a custom tokenizer.
- User wants baseline and recipe not loaded in VRAM at the same time.
- User wants Standard Eval to be a real subset benchmark, not just PPL.
- User wants eventual custom/user-added tests.
- User may want an OpenAI-compatible backend later.

## Current Recommendation

Restart from `c5666a4` and implement the next testing system in this order:

1. Native session reuse.
2. Internal small standard eval with enough samples or a curated official-subject subset.
3. Progress/cancel.
4. Official lm-eval adapter only after session reuse is proven.
5. Full official task presets later.

## Change Log After Restart

### 2026-05-28: Native ModelSession Refactor Started

Files edited:

- `native/cpp/model_surgery_runtime/src/model_surgery_runtime.cpp`

Changes made:

- Added an internal production `ModelSession` abstraction for the native llama.cpp runtime.
- `ModelSession` owns:
  - loaded `llama_model`
  - reusable `llama_context`
  - VRAM tracker
  - load timing
  - copied/converted tensor counters and byte totals
- Added session open helpers:
  - baseline/original GGUF session
  - user-copy GGUF session
  - recipe session with in-memory tensor conversion
- Refactored PPL scoring to run through a loaded session and reset llama memory between independent eval samples.
- Refactored generation/latency benchmark to run through the same loaded session after resetting context.
- Refactored existing native benchmark and recipe eval entry points to use session helpers while keeping the existing exported API unchanged.

Behavior preserved:

- No temporary GGUF is written for testing.
- Baseline and recipe compare still load sequentially, not at the same time.
- Recipe tensor conversion happens once during recipe session creation, not once per sample.
- Independent eval samples reset context/KV state.
- Existing performance and recipe metrics are kept:
  - prompt eval speed
  - token generation speed
  - TTFT
  - load time
  - elapsed time
  - peak VRAM
  - working set
  - copied tensors
  - converted tensors
  - converted bytes before/after

Validation status:

- `cargo check` passed from `src-tauri`.
- `cargo build` passed from `src-tauri`.
- `npm run build` passed from project root.
- `cargo test` passed from `src-tauri`.
- `npm test -- --runInBand` failed because `package.json` has no `test` script.
- `npm run lint` passed from project root.
- `cargo build --release` failed because `model_surgery_runtime.dll` in `src-tauri/target/release` was locked by another process:

```text
failed to copy native DLL ... model_surgery_runtime.dll ... The process cannot access the file because it is being used by another process. (os error 32)
```

No release exe was refreshed by this attempt. Close any running `model-surgery.exe` before retrying the release build.

Follow-up check found the lock owner:

```text
PID 27576 model-surgery C:\Users\Wu Family Computer\Downloads\Project 2\src-tauri\target\release\model-surgery.exe
```

After the app was closed, `cargo build --release` passed from `src-tauri`.

Updated release executable:

```text
C:\Users\Wu Family Computer\Downloads\Project 2\src-tauri\target\release\model-surgery.exe
```
