# Reproduction

Working directory: `C:/Users/public-user/Documents/DNABERT2_Finetune_HE`.

Runtime: `.tmp/privacy_runtime/python311/python.exe`, using the existing local `.venv/Lib/site-packages`. Exact Python/Torch/Transformers/GPU information is in environment.json. The local DNABERT-2 snapshot and input CSV hashes are recorded there. No package upgrade or remote model download is required for this saved environment.

Run the following sequentially for a clean execution. Existing completed stages verify or reuse their outputs. Partially started training fails rather than silently restarting with a changed state. Inspect the failure before choosing a fresh output directory. The 13-model study uses about 36 GB for 52 checkpoints plus final optimizer/RNG states, before other artifacts.

```powershell
.tmp/privacy_runtime/python311/python.exe -B -u experiments/privacy_memorization_scale_20260928/prepare.py
.tmp/privacy_runtime/python311/python.exe -B -u experiments/privacy_memorization_scale_20260928/verify.py --data-only
.tmp/privacy_runtime/python311/python.exe -B -u experiments/privacy_memorization_scale_20260928/train.py
.tmp/privacy_runtime/python311/python.exe -B -u experiments/privacy_memorization_scale_20260928/attack.py
.tmp/privacy_runtime/python311/python.exe -B -u experiments/privacy_memorization_scale_20260928/generate.py
.tmp/privacy_runtime/python311/python.exe -B -u experiments/privacy_memorization_scale_20260928/public_baseline.py
.tmp/privacy_runtime/python311/python.exe -B -u experiments/privacy_memorization_scale_20260928/analyze.py
.tmp/privacy_runtime/python311/python.exe -B -u experiments/privacy_memorization_scale_20260928/verify.py --training-replay-only
.tmp/privacy_runtime/python311/python.exe -B -u experiments/privacy_memorization_scale_20260928/attack.py --replay
.tmp/privacy_runtime/python311/python.exe -B -u experiments/privacy_memorization_scale_20260928/generate.py --replay
.tmp/privacy_runtime/python311/python.exe -B -u experiments/privacy_memorization_scale_20260928/verify.py
.tmp/privacy_runtime/python311/python.exe -B -u experiments/privacy_memorization_scale_20260928/runtime_inventory.py
.tmp/privacy_runtime/python311/python.exe -B -u experiments/privacy_memorization_scale_20260928/report.py
```

Inspect all three generated PNG figures and the numerical claims in the report. Record the actual review in `visual_review.json` (reviewed filenames and findings), then run `finalize.py` with the same Python command. This last manual review cannot be claimed automatically on a fresh rerun; finalization requires its record.

For an independent full rerun, create a new sibling folder under `experiments`, copy only `.py` files plus `config.json` and `protocol_KO.md`, and use that new folder in the commands. `core.py` resolves the project root relative to its location. Keep prior outputs intact. Source hashes of a clean rerun can include more helper files than the original freeze, because analysis helpers were implemented after the initial protocol/data/train-code freeze.

`attack.py --ready-only` and `generate.py --ready-only` evaluate checkpoints already available, for overlapping inference with training. A final call without `--ready-only` completes and checks the planned inventory. The actual execution overlapped some stages, so recorded time is not an isolated speed benchmark.

Artifacts:

- `private/`: evaluator/trainer inputs, including public background rows and synthetic secrets. This folder label describes experimental roles, not real patient data or an OS access boundary.
- `public/`: attacker challenges with no secret strings, and public vocabulary.
- `models/`: unique named-parameter checkpoints; load through `core.load` to respect tied MLM weights. Final optimizer/RNG states are saved separately.
- `predictions/`: conditional greedy/beam outputs and forward budgets.
- `generations/`: all unconditional generated strings.
- `metrics_records.json`: scoring against evaluator-only truth.
- `primary_bootstrap.json`: seed/context-pair hierarchical bootstrap summaries.
- `figures/`: PNG, PDF and SVG scientific figures.
- `verification.json`: data, checkpoint, training and generation replay evidence.
- `completion_manifest.json`: final artifact hashes.

The primary metric is hidden DNA string exact match. Token accuracy, edit similarity, unconditional substring hits, classifier metrics from older experiments, and update-based leakage are not interchangeable metrics.
