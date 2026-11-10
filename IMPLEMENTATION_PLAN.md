# Implementation plan

Two environments: **A** = Mac (CPU, development), **B** = RTX 5070 (CUDA, experiments).

| Phase | Work | Env | Status |
|---|---|---|---|
| 0 | Repository and environment audit | A | ✅ done (PROJECT_AUDIT.md) |
| 1 | Model and dataset research, verify architecture against code | A | ✅ done (MODEL_SELECTION.md, DATASET_GUIDE.md) |
| 2 | Python 3.11 environment, dependency pinning, compatibility fixes | A | ✅ done; ⏳ B: `docs/GPU_SETUP_RTX5070.md` |
| 3 | Preprocessing, model loading, inference, WER/CER | A | ✅ code + CPU smoke tests; ⏳ real checkpoint inference on B |
| 4 | Fine-tuning pipeline (freezing, AMP, accumulation, early stopping, resume) | A | ✅ code + CPU smoke test; ⏳ runs on B |
| 5 | Evaluation of A and B, comparison, charts | B | ⏳ pending GPU |
| 6 | FastAPI backend | A | ✅ done, 16 API tests |
| 7 | React frontend connected to the backend | A | ✅ done, verified in a browser (desktop + 375 px) |
| 8 | Final integration (user journey) | A/B | ✅ on A without a model; ⏳ with GPU model on B |
| 9 | Documentation and presentation material | A | ✅ docs written; ⏳ results sections after B |

## Milestones

1. ✅ A video goes in, the visual-only pipeline produces mouth crops, and a model checkpoint produces text. On the Mac this is verified with a random-weight checkpoint, end to end through the API.
2. ⏳ The same with the real pretrained checkpoint (first step on B).
3. ⏳ Experiment A `results.json`.
4. ⏳ Experiment B `results.json` and training curves.
5. ⏳ Dashboard showing the measured comparison; write-up updated.

## Next concrete tasks (on B)

Follow `CONTINUE_ON_RTX5070.md`:
1. setup
2. `gpu_check`
3. checkpoint verification
4. GRID download and preprocessing
5. memory probe
6. Experiment A
7. 20-step training test
8. Experiment B
9. final evaluation
10. commit the results
