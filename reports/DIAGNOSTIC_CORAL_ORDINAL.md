# CORAL Ordinal Image Model — Diagnostic Report

**Date**: 2026-09-28
**Subject**: `best_densenet121_ordinal.pt`, recorded val quadratic kappa = **0.094**
**Status**: **No code changed.** Diagnostics only. Scratch scripts in `/tmp/opencode/`.

---

## Verdict

**The CORAL implementation is correct. The 0.094 is not a model failure — it is a
1-epoch snapshot from a training run that was killed after epoch 1 of 100.**

`RESEARCH_AUDIT.md` originally labelled this checkpoint "FAILED/CORRUPT" and concluded
"the training diverged immediately." That conclusion was wrong. **The retraction has
since been applied** to `RESEARCH_AUDIT.md` and `IMPROVEMENTS.md` (2026-09-28); the
original wording is quoted there only to document what was corrected.

Secondary finding: `OrdinalLoss` silently discards MixUp soft labels via `argmax`,
which weakens CORAL relative to the Focal arm. That is a real (non-fatal) asymmetry,
not the cause of 0.094.

---

## 1. Pipeline trace

Dataset labels → ordinal targets → logits → sigmoid → threshold → decode → class → loss → metric

| Stage | Code | Verdict |
|---|---|---|
| Dataset labels | `prepare.py:8` `get_paths_and_labels` reads `{split}/{kl_grade}/` | OK — train `[2286,1046,1516,757,173]`, val `[328,153,212,106,27]` |
| Ordinal targets | `losses.py:42-48` `OrdinalLoss.forward` | **OK** — verified against reference |
| Logits | `image_model.py:105` `out_features = num_classes - 1` | OK — checkpoint has `(4,1024)` |
| Sigmoid | `losses.py:60` | OK |
| Thresholding | `losses.py:61` `.round()` at p=0.5 | OK — standard CORAL decode |
| Ordinal decoding | `probs.round().sum(dim=1)` | OK — round-trip verified |
| Class prediction | `trainer.py:114-117` branches on `model.ordinal` | OK |
| Loss | `F.binary_cross_entropy_with_logits` | OK — positive sign, `perfect=0.0`, `inverted=20.0` |
| Validation metric | `trainer.py:126` `cohen_kappa_score(weights='quadratic')` | OK — reproduced 0.0946 exactly from the checkpoint |

## 2. Target encoding — verified mathematically

`OrdinalLoss` builds `ordinal_labels = (targets > arange(4)).float()`, i.e. `t_kj = 1 if y > j`:

| True KL | Repo output | Reference | |
|---|---|---|---|
| 0 | `[0,0,0,0]` | `[0,0,0,0]` | OK |
| 1 | `[1,0,0,0]` | `[1,0,0,0]` | OK |
| 2 | `[1,1,0,0]` | `[1,1,0,0]` | OK |
| 3 | `[1,1,1,0]` | `[1,1,1,0]` | OK |
| 4 | `[1,1,1,1]` | `[1,1,1,1]` | OK |

## 3. Inverse decoding — verified

Perfect logits (±20) round-trip through `sigmoid` → `round` → `sum`:

| KL | logits | sigmoid | decoded |
|---|---|---|---|
| 0 | `[-20,-20,-20,-20]` | `[0,0,0,0]` | 0 |
| 1 | `[+20,-20,-20,-20]` | `[1,0,0,0]` | 1 |
| 2 | `[+20,+20,-20,-20]` | `[1,1,0,0]` | 2 |
| 3 | `[+20,+20,+20,-20]` | `[1,1,1,0]` | 3 |
| 4 | `[+20,+20,+20,+20]` | `[1,1,1,1]` | 4 |

Non-monotone profile `[0.90,0.85,0.80,0.20]` → decodes to 3, matching reference.

## 4. Real validation batch trace (16 samples, first batch)

```
true | ordinal target | raw logits                          | sigmoid                       | dec
   0 | [0, 0, 0, 0]  | [-0.079, -1.024, -1.141, -1.214]  | [0.48, 0.264, 0.242, 0.229]  |  0
   0 | [0, 0, 0, 0]  | [-0.263, -1.154, -1.490, -1.325]  | [0.435, 0.24, 0.184, 0.21]   |  0
   0 | [0, 0, 0, 0]  | [-0.243, -0.754, -1.529, -1.337]  | [0.44, 0.32, 0.178, 0.208]   |  0
   0 | [0, 0, 0, 0]  | [ 0.247, -0.662, -1.077, -0.742]  | [0.561, 0.34, 0.254, 0.323]  |  1
   0 | [0, 0, 0, 0]  | [ 0.103, -0.994, -1.169, -0.964]  | [0.526, 0.27, 0.237, 0.276]  |  1
   0 | [0, 0, 0, 0]  | [ 0.034, -0.523, -1.120, -0.849]  | [0.508, 0.372, 0.246, 0.3]   |  1
```

- batch predictions `[9,7,0,0,0]` vs truth `[16,0,0,0,0]`
- **monotone-decreasing sigmoid rows: 0/16 (0.0%)**
- mean sigmoid per task `[0.498, 0.328, 0.217, 0.270]`

Full val set (n=826), recomputed from the checkpoint:

```
quadratic kappa = 0.0946   (matches recorded 0.0946)
exact accuracy  = 0.3184   MAE = 1.069   within-1 = 0.7107
pred dist = [458, 358, 8, 2, 0]     true dist = [328, 153, 212, 106, 27]
```

The model can only emit 0 and 1. It never predicts KL≥2.

## 5. The head is essentially untrained

```
classifier.net.5.weight  absmax = 0.0341     classifier.net.5.bias  absmax = 0.0329
nn.Linear(1024, 4) PyTorch init bound       = 0.03125
classifier.net.1.weight  absmax = 0.0330     (also at init)
```

The entire head — both the 1024-unit layer and the 4-unit output — sits at its
initialization magnitude. Nothing was learned.

## 6. Root cause: the run died after epoch 1

`logs/compare_models.log` — the ordinal run is the last entry and stops dead:

```
2026-09-08 18:09:57 | Epochs: 100 | Models: densenet121 | Ordinal: True | Patience: 15
2026-09-08 18:10:00 | Parameters: 8,009,604
2026-09-08 18:11:17 | Epoch  1 | 77s | Train Loss: 0.6116 | Kappa: 0.0946 | EMA Kappa: 0.0172
2026-09-08 18:11:17 |   -> Saved best (kappa=0.0946)
<end of log — no epoch 2>
```

Corroborating evidence:
- `best_densenet121_ordinal.json` records `"epoch": 1`
- `models/checkpoint_densenet121_ordinal.pt` **does not exist** — with
  `CHECKPOINT_INTERVAL=5` the first resume point would be epoch 5, so the process
  was terminated in epochs 2–4
- no MLflow run exists for the 2026-09-08 ordinal attempt

`best_kappa` is a **max over epochs**, so the first epoch's value was written to
disk and never beaten. The file is a valid, correctly-shaped, untrained model.

## 7. Controlled comparison — CORAL vs 5-class

Identical dataset, split, backbone (`densenet121`), augmentation, sampler, optimizer
(AdamW lr 1e-4, wd 1e-4), LR schedule, batch 32, clip 1.0, seed 42. Only the head
width and loss differ.

| Arm | epochs | best quadratic kappa | acc | MAE | within-1 |
|---|---|---|---|---|---|
| **CORAL (ordinal)** | 6 | **0.6827** | 0.5303 | **0.581** | 0.895 |
| Standard 5-class (Focal) | 6 | 0.6327 | 0.4613 | 0.660 | 0.885 |

Per-epoch CORAL: `0.2912 → 0.5428 → 0.6779 → 0.6827 → 0.6812` (best 0.6827)
Per-epoch standard: `0.3150 → 0.4281 → 0.5246 → 0.6255 → 0.6327 → 0.6114` (best 0.6327)

**Interpretation**: CORAL is demonstrably *learning* — its validation kappa rises
steadily and it was not inferior to the Focal arm on the ordinal metrics tested at
this budget. **This is not a superiority claim.** The run was truncated to 6 epochs
for runtime, is not converged, uses a single seed, and the two arms were not given
equivalent augmentation (see #10 and #11 below). The full 100-epoch budget was not
tested, and the arms may converge differently. Re-run to completion, across seeds,
before drawing any conclusion in either direction.

## 8. Audit of the 12 suspected causes

| # | Suspected cause | Verdict |
|---|---|---|
| 1 | Incorrect target encoding | **No** — matches reference exactly |
| 2 | Incorrect thresholding | **No** — `.round()` at 0.5 is standard CORAL |
| 3 | Incorrect class ordering | **No** — KL 0..4 ascending, `t_kj = 1 if y>j` |
| 4 | Incorrect loss sign | **No** — `perfect=0.0`, `random=0.79`, `inverted=20.0` |
| 5 | Incorrect cumulative prob interpretation | **No** — `ordinal_to_probs` differences are right |
| 6 | Wrong logits/output dim | **No** — `(4,1024)` as designed for K=5 |
| 7 | Train/eval label mismatch | **No** — same `get_paths_and_labels`; kappa reproduced 0.0946 exactly |
| 8 | Checkpoint loading issue | **No** — raw `OrderedDict`, `_infer_ordinal` → `True`, loads clean |
| 9 | Metric calculation issue | **No** — `cohen_kappa_score(weights='quadratic')` correct; base-rate kappa is 0.0 |
| 10 | Class-weight issue | **Real but non-fatal** — see below |
| 11 | Focal × CORAL interaction | **Real but non-fatal** — see below |
| 12 | Augmentation issue | **No** — shared by both arms; controls cleanly |

### 10. Class weighting is asymmetric

`compare_models.py:89-93`:
- standard → `FocalLoss(alpha=[0.506, 1.105, 0.762, 1.527, 6.68], label_smoothing=0.1)`
- CORAL → `OrdinalLoss(num_classes=5)` — **no alpha, no smoothing**

The KL4 class weight is 6.68, the largest by far. The CORAL arm has no compensating
reweighting. Also, `losses.py:52` applies a per-grade weight uniformly to all four
binary tasks, so a KL4 sample up-weights "is grade>3" as heavily as "is grade>0" —
wrong for a cumulative encoding. This is an unresolved confound, and one reason the
6-epoch result above cannot be read as a CORAL-vs-Focal verdict.

### 11. MixUp silently discards the ordinal signal

`compare_models.py:76-77` wraps the train set in `MixUpDataset` when
`MIXUP_ALPHA=0.4 > 0`, which returns soft 2-D one-hot targets. `losses.py:44-45`:

```python
if targets.ndim == 2:
    targets = targets.argmax(dim=1)   # soft mixup target -> hard label
```

A `[0.5, 0, 0, 0, 0.5]` KL0/KL4 blend is discarded in favour of whichever class
wins the argmax. `FocalLoss._soft_focal_loss` handles soft targets natively, so the
two arms do **not** receive equivalent augmentation. This needs a dedicated
fairness/ablation study before the two arms are compared.

### Also noted (not implicated)

`scripts/train_xray_multitask.py:171` writes to `best_{model}_multitask.pt`
regardless of `--ordinal`, so a multitask CORAL run would overwrite the
non-ordinal multitask checkpoint (`best_densenet121_multitask.json` currently
records `"ordinal": false`).

---

## 9. Recommended corrections

1. **Retract the audit claim.** *(DONE 2026-09-28)* `RESEARCH_AUDIT.md` and `IMPROVEMENTS.md` called the checkpoint corrupt and said training diverged. It did not diverge; the run was killed at epoch 1. Both documents now describe it as an interrupted/incomplete epoch-1 artifact, and the CORAL implementation and ordinal encoding as validated correct. `best_densenet121_ordinal.pt` should still be deleted or quarantined so it cannot be loaded as a trained model.
2. **Do not patch the CORAL implementation.** The math is correct.
3. Optional real improvements, in priority order: preserve MixUp soft targets in
   `OrdinalLoss` by using `target @ ordinal_pattern` instead of `argmax`; give the
   CORAL arm class weights and a per-task weighting scheme.
4. Re-run the full 100-epoch comparison before claiming CORAL is superior — 6 epochs
   is not a converged result, and the arms may cross over later.
5. Fix the multitask checkpoint filename to include the ordinal flag.

## 10. Reproduction

- `/tmp/opencode/diag_coral_math.py` — target/decode math vs reference
- `/tmp/opencode/diag_coral_trace.py` — real-batch pipeline trace + kappa recomputation
- `/tmp/opencode/controlled_compare.py` — CORAL vs 5-class, identical config

No repository file was modified. `git status` shows only pre-existing changes
(`scripts/train_fusion.py`, `src/kneevision/evaluation/report.py`, `streamlit_app.py`).
