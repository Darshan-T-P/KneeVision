# X-ray Model Training Strategy

Scope: **the image branch only** (`src/kneevision/models/image_model.py`). The
clinical-text and fusion branches are already strong (87.4% / 88.9% accuracy,
0.953 / 0.959 quadratic kappa — see main README, Phase 4/5) because they were
enriched with real radiographic findings; that lever is used up. The image
branch is the one place left with real, unexploited room to improve, and it's
constrained to the same 8,260 Kaggle X-rays — no new labeled images are
available (see README Phase 4 note on the OAI image-release gap). Every
experiment below works within that constraint.

## Current baselines (official, same 1,656-image test set)

| Model | Accuracy | Quadratic Kappa | Macro F1 | KL3 Recall | Source |
|---|--:|--:|--:|--:|--:|
| DenseNet121 (5-class, champion) | 60.3% | 0.792 | 0.592 | 27% | `reports/evaluate_fusion/metrics.json` |
| DenseNet121 + multitask aux heads | 59.5% | 0.801 | 0.621 | 43% | ad hoc eval, `models/best_densenet121_multitask.pt` |
| EfficientNet-B4 (5-class) | — | 0.640 (val) | — | — | `models/best_efficientnet-b4.json` |
| DenseNet121+ConvNeXt-Small (binary ensemble) | 86.2% | 0.757 (val) | — | n/a | `reports/binary_dedicated/` |

**The standing problem, unchanged across every variant tried so far:** KL1 is
the weakest class everywhere (F1 0.35–0.40), and KL3 recall is badly
asymmetric relative to precision (a model that's `unwilling` to call KL3
unless very sure). The multitask experiment (already committed) is the one
real, verified win: +0.03 macro F1, KL3 recall 27%→43%, at a small cost to
overall accuracy (-0.8pp) and KL0/KL2 precision. It stays as a checkpoint,
not yet promoted to champion.

## Evaluation protocol (applies to every candidate below)

1. Evaluate every candidate on the exact same held-out test set (1,656
   images) via `scripts/evaluate.py <name>` or `scripts/evaluate_fusion.py`
   — never compare numbers pulled from different scripts/runs (an earlier
   mistake in this project: `evaluate.py`'s TTA-enabled numbers and
   `evaluate_fusion.py`'s no-TTA numbers disagreed for the *same* checkpoint
   until this was noticed).
2. **Primary metric: quadratic Cohen's kappa** (clinically-weighted — an
   off-by-one error costs less than off-by-three). **Secondary: macro F1**
   (equal weight per class, so it can't be gamed by nailing KL0).
   **Tracked but not optimized directly: per-class KL1 F1 and KL3 recall**
   — the two specific known failure modes.
3. **Promotion rule:** a candidate replaces `models/best_densenet121.pt` as
   champion only if it beats 0.792 kappa *and* doesn't drop macro F1 below
   0.592. Otherwise it's kept as a separate checkpoint (`best_<name>.pt`,
   never overwriting the champion) and documented as a partial win, same as
   the multitask variant today.
4. Every run logs to MLflow (`mlflow.db`) so results are comparable later
   without re-running anything.

## Candidates, in priority order

Ordered by expected-impact-per-hour-of-GPU-time, cheapest/lowest-risk first.

### 1. Ordinal (CORAL) loss for the main 5-class model
- **Hypothesis:** the champion DenseNet121 is trained with Focal loss
  (`ordinal: false` per its own metadata) — a plain multi-class classifier
  with no notion that KL grades are ordered. `OrdinalLoss` already exists
  and is used elsewhere (fusion head, multitask variant), but never for the
  standalone champion. Ordinal training tends to reduce the *worst* errors
  (predicting KL0 when it's actually KL4) even when overall accuracy is flat.
- **Method:** `uv run python scripts/compare_models.py --models densenet121
  --ordinal`. `run()` already accepted an `ordinal` parameter, but the CLI
  never exposed it — added the `--ordinal` flag (and fixed a dormant bug
  found while doing so: `OrdinalLoss` was hardcoded to `num_classes=5`
  regardless of `--binary`, which would have silently broken an
  ordinal+binary combination had one ever been run).
- **Cost:** ~1 full training run, same order of time as any other
  `compare_models.py` run (historically under an hour on this GPU for
  DenseNet121).
- **Risk:** low — doesn't touch data pipeline, only the loss/decoding.

### 2. Higher input resolution (320px)
- **Hypothesis:** osteophytes and joint-space narrowing are fine bone-edge
  details; 224px may be discarding resolution that matters. Untested in
  this project. Common finding in the KL-grading literature.
- **Method:** `IMAGE_SIZE=320 uv run python scripts/compare_models.py
  --models densenet121`. Verified: `settings.py` reads `IMAGE_SIZE` from
  the environment at import time, and `transforms.py`'s module-level
  `train_transform`/`val_transform`/`minority_transform` are built from it
  — the env var genuinely controls resolution end to end, no code change
  needed. **Gotcha:** `IMAGE_SIZE` must be set identically for the eval
  command too (`IMAGE_SIZE=320 uv run python scripts/evaluate.py
  densenet121_320...`), or train/eval resolution will silently mismatch.
- **Cost:** similar per-epoch cost to #1, likely slower per step (more
  pixels) and higher VRAM (watch the 6GB budget on this GPU).
- **Risk:** low-medium — batch size may need to shrink to fit VRAM at
  320px; if so, `SAMPLER_POWER`/`BATCH_SIZE` interact, re-check they're
  still sensible.

### 3. Ordinal + multitask combined
- **Hypothesis:** the two working levers found so far (#1's ordinal
  decoding, and the already-committed multitask auxiliary supervision) are
  independent mechanisms — one changes the loss/decision boundary, the
  other adds regularizing signal from real radiographic sub-components.
  Worth trying together rather than assuming they don't compose.
- **Method:** `scripts/train_xray_multitask.py` already accepts
  `--ordinal`; run it and compare against both baselines above.
- **Cost:** same as the original multitask run (~45 min on this GPU per
  the last run's `Time: 2738s` log).
- **Risk:** low — purely additive to already-working code paths.

### 4. ViT-B/16 or Swin-T as an ensemble partner
- **Hypothesis:** every trained backbone so far is a CNN (DenseNet,
  EfficientNet, ConvNeXt). `vit_b_16`/`swin_t` are registry-supported
  (`AVAILABLE_MODELS`) but have zero trained checkpoints anywhere in this
  project. A transformer's different inductive bias is more likely to add
  genuine ensemble diversity than another CNN.
- **Method:** `uv run python scripts/compare_models.py --models vit_b_16`,
  then `scripts/evaluate.py densenet121 vit_b_16` for the ensemble number.
- **Cost:** highest of the four — ViT-B/16 is a larger model than
  DenseNet121; expect a longer per-epoch time and confirm it fits in 6GB
  VRAM at batch size 32 before committing to a full run (drop batch size
  if not).
- **Risk:** medium — first-time use of this backbone in this project, more
  likely to surface an unexpected shape/compat issue than the others.

### Deliberately out of scope right now
- **Two-stage hierarchical (binary screen → severity-only classifier):**
  the most invasive redesign (new training pipeline, new inference-time
  branching logic in every consumer: Streamlit, API, fusion). Worth
  revisiting only if #1–#4 fail to move KL1/KL3, since it's a structural
  change rather than a training-time one.
- **New image data:** confirmed unavailable (OAI's free release is
  tabular-only; see README Phase 4). Not re-litigated here.

## Execution order

Run #1 → evaluate → #2 → evaluate → #3 → evaluate → #4 → evaluate, updating
the results table in this doc after each one (not just at the end), so a
partial run always leaves the doc in a consistent, truthful state. Stop
early and reassess if a candidate clearly underperforms rather than running
the full list mechanically.
