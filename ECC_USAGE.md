# ECC usage

ECC **2.2.3** (agent tooling plugin).

- At the first audit, ECC was **not installed**: no plugin directory and no `python-patterns` skill.
- It became available during the session.

This file records only what actually happened. ECC was not modified.

## Inventory inspected

ECC exposes roughly 75 agents, 90 slash-command skills and about 300 further skills. They
cover many languages and domains. The ones relevant to this project, and what was done
with each:

| Capability | Type | Used? | How / why |
|---|---|---|---|
| `python-patterns` | skill | **yes** | Loaded before writing ML/backend code. Applied: type hints, frozen dataclasses (`ManifestEntry`, `VideoInfo`, `ErrorCounts`), a custom exception hierarchy (`VideoDecodeError` → `VideoTooLongError`, `MouthCropError`, `CheckpointError`, `DeviceUnavailableError`), no bare `except`, `pathlib` |
| `pytorch-patterns` | skill | **yes** | Device-agnostic code with explicit device policy, seeding, `torch.load(weights_only=True)`, `model.eval()` + `no_grad` for inference, AMP with GradScaler only for fp16, complete resumable checkpoints |
| `fastapi-patterns` | skill | **yes** | App factory with lifespan, `pydantic-settings` config, typed response models, thin routes over a service layer, TestClient tests |
| `frontend-design-direction` | skill | **yes** | Wrote the design contract (purpose, audience, tone, memorable detail = the real mouth-crop strip) and designed the content states (loading, empty, error, partial, success, permission). The bundled evidence-checker script was **not** run |
| `ecc:mle-reviewer` | agent | **yes** | Read-only review of `ml/` and the GPU probe |
| `ecc:fastapi-reviewer` | agent | **yes** | Read-only review of `backend/` |
| `ecc:react-reviewer` | agent | **yes** | Read-only review of `frontend/src` |
| GateGuard (`pre:bash` / `pre:edit-write` fact-forcing hooks) | hook | active | Required stating the purpose, callers and data schema before each first file creation per directory and before destructive shell commands. All prompts were answered; the hook was **not** disabled |
| StrategicCompact, scope-warning, cost-notice hooks | hooks | active | Informational only |
| `python-reviewer`, `pytorch-build-resolver`, `tdd-guide`, `e2e-runner`, `security-reviewer`, `doc-updater`, `/code-review`, `/plan` | various | no | Not needed: the three domain reviewers covered review; tests were written alongside the code; no build failures needed a resolver |

## Review results and actions

The three reviewer agents produced 29 findings. Every high and medium finding was fixed
and retested.

| Reviewer | Finding (severity) | Action |
|---|---|---|
| mle | Batches single-speaker and identical every epoch, from a stable sort over speaker-ordered equal lengths (HIGH) | random tie-break per epoch; test `test_sampler_mixes_equal_length_clips_across_speakers_and_epochs` |
| mle | No NaN/Inf guard under bf16 (MED) | non-finite steps skipped and counted; abort after 50 |
| mle | Lossy resume (patience, scaler, partial epochs, config drift) (MED) | all saved; `last.pth` only at full epochs; config mismatch refuses resume |
| mle | Scheduler horizon used `ceil` (MED) | `floor` |
| mle | Frozen BN statistics could drift when `freeze_batchnorm: false` (MED) | frozen BN always in eval mode |
| mle | Preprocessing failures silently shrink the test set (MED) | per-split failure counts in `prep_report.json`; `excluded_by_preprocessing` in results |
| mle | Dead config keys; evaluating with the training config scores the base model (LOW) | keys removed; `finetuned_grid_eval.yaml`; `evaluate` rejects training configs |
| fastapi | Blocking, unbounded decode in an async route (HIGH) | validation in a thread pool; resample while decoding; 960 px cap for uploads; output-frame limit → `video_too_long` |
| fastapi | Temp-file leaks on non-`ApiError` paths (HIGH) | `BaseException` cleanup; startup sweep; queued uploads removed on shutdown |
| fastapi | A model-load error other than `CheckpointError` leaves the state "loading" forever (HIGH) | catch-all → state `error` with a logged traceback |
| fastapi | Eviction of live jobs, unbounded queue, torn reads (MED) | evict finished jobs only; 429 `queue_full`; per-job lock, status written last |
| fastapi | Size limit only enforced after spooling the body (MED) | middleware rejects on `Content-Length`, and 503 before body parsing |
| fastapi | 404/405 outside the envelope; non-serialisable validation errors (MED) | `StarletteHTTPException` handler; `jsonable_encoder` |
| fastapi | Internal exception text returned to clients (LOW) | generic message; details only in the server log |
| react | Stale submit could attach a result to another file (HIGH) | sequence-guarded submit; input ignored while busy |
| react | Camera left on after unmount or a pending `getUserMedia`; recorder not stopped (HIGH) | mounted guard, stream replacement, recorder stopped and detached on unmount, "turn camera off" button |
| react | All camera errors shown as "blocked"; unhandled MediaRecorder errors; empty clips (HIGH) | NotAllowed / NotFound / other distinguished; `onerror`; empty clips ignored |
| react | Non-null assertions on possibly null metrics (HIGH) | comparison shown only when every number exists |
| react | `useApi` out-of-order and post-unmount updates; `useJob` stops on one blip (MED) | sequence + alive guards; retry with backoff, 404 final |
| react | Tabs without a tablist keyboard model; live regions mounted late (MED) | `aria-pressed` toggle buttons; one persistent `role=status` region |
| react | Clipboard rejection, object URL in `useMemo`, misleading "skipped" text, fixed "8 frames" alt (LOW) | all fixed |

Not changed, and why:
- **No CORS headers on 500 responses:** the browser goes through the Vite same-origin proxy.
- **Module-level `app` instance:** kept for the simple `uvicorn backend.app.main:app` command.
- **Weight decay applied to norm/bias parameters (LOW):** kept as in upstream Auto-AVSR.
- **The memory probe slightly underestimates the real peak (LOW):** documented with ~2 GB headroom in the GPU guide.

## Not available or not used

- **ECC MCP tools** (`chrome-devtools`): the built-in browser pane was used for visual verification instead.
- **No ECC commands** that need external services or accounts (Codex, Jira, GitHub epics) were run.
