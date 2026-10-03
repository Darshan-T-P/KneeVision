# KneeVision++ Research Methodology Audit

**Date**: 2026-09-28  
**Auditor**: Antigravity AI (automated code + data inspection)  
**Scope**: Full repository audit — code, data, experiments, claims  
**Revision**: 2026-09-28 — CORAL findings corrected after dedicated diagnostic
(see "Correction Notice" below and `reports/DIAGNOSTIC_CORAL_ORDINAL.md`).

---

## Correction Notice — CORAL Ordinal Model (revised 2026-09-28)

**The original audit incorrectly classified the CORAL ordinal image model as
"broken" / "corrupt" and concluded that training "diverged immediately". Both
statements are withdrawn.**

A dedicated diagnostic audit (`reports/DIAGNOSTIC_CORAL_ORDINAL.md`) established that:

1. **The CORAL implementation is correct.** The loss, thresholding, class ordering,
   output dimension, checkpoint loading and metric calculation were each verified
   against a closed-form reference.
2. **The KL → ordinal target encoding is correct** and was verified exactly:

   | True KL | Ordinal target | Expected |
   |---|---|---|
   | 0 | `[0,0,0,0]` | `[0,0,0,0]` |
   | 1 | `[1,0,0,0]` | `[1,0,0,0]` |
   | 2 | `[1,1,0,0]` | `[1,1,0,0]` |
   | 3 | `[1,1,1,0]` | `[1,1,1,0]` |
   | 4 | `[1,1,1,1]` | `[1,1,1,1]` |

   Inverse decoding (`sigmoid` → `round` → `sum`) was verified to recover all five
   grades from perfect logits, including a non-monotone probability profile.
3. **`best_kappa = 0.0946` is the product of an interrupted run, not a failed model.**
   The training process terminated after epoch 1 of a planned 100 epochs. Because
   `best_kappa` is a running maximum over epochs, the epoch-1 value was written to
   disk and never superseded. The checkpoint is correctly shaped and loads cleanly;
   its head weights remain at initialization magnitude, confirming it is incomplete
   rather than divergent.
4. **A controlled 6-epoch experiment showed CORAL learning correctly**, but it is
   truncated and **no claim of CORAL superiority is made or implied** (see
   "Open Question 2" below).

The original diagnostic evidence is preserved in full in
`reports/DIAGNOSTIC_CORAL_ORDINAL.md` and is not retracted — only its
interpretation of the checkpoint status is corrected here.

---

## Summary Rating

| Category | Status |
|---|---|
| Dataset splitting (patient-level) | ✅ IMPLEMENTED correctly |
| CORAL ordinal loss | ✅ VALIDATED — implementation and target encoding verified correct |
| CORAL ordinal image training run | ⚠️ INCOMPLETE — interrupted after epoch 1 of 100 |
| Ordinal loss vs Focal fairness (MixUp, class weights) | ⚠️ ASYMMETRIC — ablation not yet run |
| Auxiliary OARSI heads | ✅ IMPLEMENTED (real OAI data) |
| Multimodal pairing (genuine) | ✅ VERIFIED — 100% match rate |
| Fusion evaluation pipeline | ✅ IMPLEMENTED |
| XAI (Grad-CAM, Score-CAM, LIME) | ✅ IMPLEMENTED |
| RAG pipeline | ✅ IMPLEMENTED |
| OARSI → KL label leakage | ⚠️ CRITICAL FINDING — documented below |
| KL3/KL4 perfect scores | ⚠️ SUSPICIOUS — likely from OARSI leakage, not image learning |
| Calibration evaluation | ❌ NOT IMPLEMENTED |
| MAE / Within-1 / Within-2 metrics | ❌ NOT REPORTED |
| Ordinal safety metrics | ❌ NOT IN evaluate_fusion.py |
| XAI quantitative validation | ❌ NOT IMPLEMENTED |
| RAG retrieval evaluation | ❌ NOT IMPLEMENTED |
| External validation dataset | ❌ NOT IMPLEMENTED |
| Reproducibility seeds | ⚠️ PARTIALLY — set_seed used in multitask, not in all scripts |

---

## Phase 1 — Claim-by-Claim Verification

### ✅ Image Dataset (Kaggle Knee OA)
**VERIFIED**: 8,260 images across train/val/test directories.
- Train: 5,778 | Val: 826 | Test: 1,656
- Distributions match paper claims exactly.

### ✅ Patient-Level Splitting
**VERIFIED**: Zero patient leakage across all splits.
- Filenames encode patient ID + side (e.g., `9911788L.png`).
- Kaggle dataset = OAI patient cohort (100% patient ID overlap).
- No patient appears in more than one split.
- 4,130 unique patients, each appearing in exactly one split.

### ✅ DenseNet121 Backbone + ImprovedHead
**VERIFIED**: Correctly implemented. CORAL ordinal head (`out_features = num_classes - 1 = 4`).
- Backbone registry with 11 supported architectures is present.
- `extract_features()` returns raw backbone features for fusion.

### ✅ CORAL Ordinal Loss — VALIDATED
**VERIFIED CORRECT**: The CORAL implementation was validated element by element in
`reports/DIAGNOSTIC_CORAL_ORDINAL.md`.
- `ordinal_to_class()` uses `probs.round().sum()`, a valid CORAL decoding; round-trip verified for all five KL grades.
- `ordinal_to_probs()` correctly converts cumulative sigmoids to a valid grade distribution.
- **Target encoding verified** against the CORAL reference `t_kj = 1 if y > j` for KL 0–4 (table in the Correction Notice).
- Loss sign verified: perfect logits → 0.0, random → 0.79, inverted → 20.0.
- Output dimension verified: `(4, 1024)` for `num_classes=5`, as designed.
- Checkpoint loading verified: raw `OrderedDict`, `_infer_ordinal()` → `True`, loads without error; recorded κ=0.0946 was reproduced exactly from the checkpoint.

- **NOTE (was previously "BUG FOUND")**: `best_densenet121_ordinal.pt` records `best_kappa=0.0946`. This is an **interrupted/incomplete training artifact**, not a corrupt or divergent model — the run stopped after epoch 1 of 100. The original "essentially untrained/corrupt" wording is withdrawn.
- The image-only model used in fusion evaluation is `best_densenet121.pt` (standard, non-ordinal, kappa=0.775 on val).

### ✅ BioClinicalBERT Clinical Text Model
**VERIFIED**: Uses `emilyalsentzer/Bio_ClinicalBERT`. Direct [CLS] token → Linear classifier. `extract_features()` returns 768-dim embedding for fusion.

### ✅ MultimodalFusionModel
**VERIFIED**: Late fusion. Encoders frozen during fusion training. Fusion head: `Linear(1792→512)→BN→ReLU→Linear(512→4)`.
- `best_fusion.pt` (448MB) exists and includes both sub-models frozen weights.

### ✅ Data Pairing (Genuine)
**VERIFIED**: All 1,656 test images match OAI clinical records by patient ID + side with 100% match rate. `allow_fallback_text=True` is never invoked in practice (all records matched).

### ✅ Augmentation Pipeline
**VERIFIED**: Implemented as described. However:
- **CONCERN**: `ColorJitter(saturation=0.2, hue=0.1)` applied on minority-class augmentation path. X-rays are grayscale. After `convert("RGB")`, hue/saturation shifts are meaningless but also harmless.
- `RandomGrayscale(p=0.1)` in minority transform: no-op on already-grayscale images.
- Horizontal flip for knee X-rays: clinically concerning (flips left knee to right knee presentation) but commonly used and debated in the literature.

### ✅ Auxiliary OARSI Heads (Multitask)
**VERIFIED**: `KneeXRayAuxDataset` correctly loads OAI features from CSV. Auxiliary heads trained on real OAI per-compartment grades. `AUX_IGNORE_INDEX = -100` used for missing fields.

### ✅ XAI (Grad-CAM, Score-CAM, LIME)
**VERIFIED**: All three implemented. Hook on `backbone.features.denseblock4`. Heatmaps normalized and alpha-composited.
- **CONCERN**: Score-CAM does two forward passes per channel (slow for 1024 channels).
- **GAP**: No quantitative XAI evaluation exists (insertion/deletion, localization).

### ✅ RAG Pipeline
**VERIFIED**: TF-IDF + KL-grade boost + Ollama (Llama 3.2) with graceful fallback. Guidelines corpus exists in `data/guidelines/`. Clear "not medical advice" disclaimers throughout.

### ✅ Tests
**VERIFIED**: 199 tests, all passing. Good coverage of individual components.

### ❌ EMA During Fusion Training
**NOT IMPLEMENTED in `train_fusion.py`**: The image training scripts use EMA but `train_fusion.py` does not.

### ❌ Missing Metrics
`compute_metrics()` does NOT compute:
- MAE (mean absolute error)
- Within-1 accuracy
- Within-2 accuracy
- Calibration metrics (ECE, Brier score)

---

## Phase 2 — Data Validity Findings

| Metric | Value |
|---|---|
| Total image records | 8,260 |
| Unique patients (Kaggle) | 4,130 |
| Unique knees | 8,260 |
| OAI unique patients (CSV) | 4,506 |
| Kaggle ∩ OAI patients | 4,130 (100%) |
| Test set multimodal matches | 1,656 / 1,656 (100%) |
| Cross-split patient leakage | 0 patients |
| Cross-split knee leakage | 0 knees |
| KL label agreement (Kaggle vs OAI) | 99.4% (51/8,260 minor disagreements) |

**Finding**: The Kaggle Knee OA dataset is derived from OAI subject data. Patient IDs are OAI participant IDs. The 51 label disagreements (0.6%) are likely rounding or baseline-visit discrepancies.

---

## Phase 3 — OARSI / KL Label Leakage (CRITICAL)

### The Core Issue

The Kellgren-Lawrence (KL) grading system is **operationally defined** as a function of:
- Joint space narrowing (JSN)
- Osteophytes
- Subchondral sclerosis
- Bone attrition

These are exactly the OARSI per-compartment fields in `kxr_sq_bu00.txt` that are encoded in the clinical text.

**This means**: The BioClinicalBERT branch is learning to decode the KL grade from its defining radiographic components — not from patient-reported outcomes or independent clinical information.

### Evidence from OARSI Coverage

| OARSI Field | Test Coverage |
|---|---|
| jsn_m, jsn_l | 100% (2,478/2,478) |
| osteophyte_m, osteophyte_l | 54% (1,347/2,478) |
| sclerosis_m, sclerosis_l | 33% (829/2,478) |
| attrition_m, attrition_l | 33% (829/2,478) |

### OARSI-KL Monotonic Relationship

| KL Grade | Mean JSN_M | Mean Osteophyte_M | Mean Sclerosis_M |
|---|---|---|---|
| KL 0 | 0.00 | 0.02 | 0.00 |
| KL 1 | 0.42 | 0.44 | 0.02 |
| KL 2 | 0.53 | 1.14 | 0.37 |
| KL 3 | 1.59 | 1.85 | 1.26 |
| KL 4 | 2.10 | 2.67 | 1.81 |

These values are nearly perfectly monotonic — exactly as expected since KL IS defined by these components.

### Provenance Assessment

The code comment in `compose_clinical_report()` states:
> "The overall KL grade is deliberately NOT included in the text: it is the label the model must predict"

This is technically correct — the raw KL grade string is not in the text. However, **the components that jointly define KL grade ARE present**, and they come from the **same annotation session** (`kxr_sq_bu00.txt`) as the KL label itself.

### Classification of the Leakage Type

This is **definitional co-occurrence**, not engineering label leakage (where the label string appears in the feature). The OARSI components and KL grade are from the same radiologist read. This is a **fundamental experimental design issue** that must be disclosed clearly in the paper.

### Impact on Results

- Clinical text model κ=0.953 is explained by this co-occurrence.
- Multimodal fusion κ=0.959 adds only marginal value (+0.006) because the text branch already has near-ceiling performance.
- KL3/KL4 fusion recall = 99.6%/100%: these grades have strong OARSI signals (JSN≈2-3, osteophytes≈2-3).
- KL1 recall = 56.4%: KL1 is ambiguous even in OARSI reads (JSN≈0.42, osteophytes≈0.44 — borderline values).

### What Is NOT Leakage

- The OARSI fields were read **independently from the same X-ray** by a different abstraction process than KL grading.
- No patient-reported symptoms (WOMAC) directly determine KL grade.
- This is genuinely how clinical radiologists work: JSN + osteophytes → KL grade.

---

## Phase 4 — Image Model Audit

### Checkpoint Status

| Checkpoint | Best Kappa (val) | Training Completed | Status |
|---|---|---|---|
| `best_densenet121.pt` | 0.775 | 30 epochs, full run | ✅ Standard cross-entropy + Focal loss |
| `best_densenet121_multitask.pt` | 0.799 | 36 epochs, early-stopped (best at 21) | ✅ OARSI auxiliary heads added (non-ordinal) |
| `best_densenet121_ordinal.pt` | 0.0946 | **1 of 100 — interrupted** | ⚠️ INCOMPLETE — not corrupt, not divergent |

### Root Cause of the 0.0946 Ordinal Checkpoint

**Revised finding (2026-09-28).** The original audit reported that "the ordinal
(CORAL) image model failed to train" and that training "diverged immediately."
Both claims are **withdrawn**. The evidence shows an externally interrupted run:

- `models/best_densenet121_ordinal.json` records `"epoch": 1`.
- `logs/compare_models.log` ends immediately after the epoch-1 line:
  ```
  2026-09-08 18:09:57 | Epochs: 100 | Models: densenet121 | Ordinal: True | Patience: 15
  2026-09-08 18:11:17 | Epoch  1 | 77s | Train Loss: 0.6116 | Kappa: 0.0946 | EMA Kappa: 0.0172
  2026-09-08 18:11:17 |   -> Saved best (kappa=0.0946)
  <no further epochs logged>
  ```
- `models/checkpoint_densenet121_ordinal.pt` does not exist. With
  `CHECKPOINT_INTERVAL=5` the first resume point would be epoch 5, so the process
  was terminated during epochs 2–4.
- No MLflow run exists for the 2026-09-08 ordinal attempt.
- Head weights are at initialization magnitude
  (`classifier.net.5.weight` absmax 0.0341 vs PyTorch init bound 0.03125),
  confirming the network was interrupted before it learned, not diverged to NaN.

Because `best_kappa` is a **running maximum across epochs**, the epoch-1 value was
persisted and never superseded. The file is a valid, correctly-shaped,
**incomplete** model.

**Consequence for reporting**: `best_densenet121_ordinal.pt` must not be cited as an
ordinal-image-model result. There is currently **no completed CORAL image model** in
the repository. All reported image-branch results come from `best_densenet121.pt`
(standard, non-ordinal, κ=0.775).

### CORAL Implementation Status

The implementation itself is **validated and correct** — see the Correction Notice
and `reports/DIAGNOSTIC_CORAL_ORDINAL.md`. The outstanding gap is a *completed
training run*, not defective code.

### Image Model Used in Fusion

`evaluate_fusion.py` loads `best_densenet121.pt` (non-ordinal, kappa=0.775 val). The fusion head is ordinal (CORAL), but the image encoder it's built on was not trained with ordinal loss.

### Augmentation Concerns

- `ColorJitter(saturation, hue)` on minority class: **effectively no-op** for grayscale X-rays, but harmless.
- Horizontal flip: anatomically flip left/right, which is a recognized but debated augmentation for knee X-rays.
- `RandAugment(N=2, M=9)`: may introduce posterization, solarization, equalization — not all appropriate for X-rays. **Controlled ablation not performed.**

---

## Phase 5 — KL1 Failure Analysis

From stored metrics (image-only model on test set):
- KL1 precision = 0.321, recall = 0.456, F1 = 0.377
- KL1 AUC = 0.719 (lowest of all grades)

From fusion model:
- KL1 recall = 0.564 — still lowest by far
- Primary confusion: KL1→KL0 (predicted normal when mild OA present)

**Root cause hypothesis**: KL1 ("doubtful OA") is intrinsically ambiguous even for human radiologists. The OARSI JSN_M for KL1 averages only 0.42 (near-zero), osteophytes 0.44 — borderline values that text cannot cleanly separate from KL0.

---

## Phase 9 — RAG Evaluation Gap

No formal evaluation exists for:
- Retrieval precision/recall
- Groundedness of LLM output
- Hallucination rate
- Citation accuracy

The RAG system falls back gracefully to raw excerpts when Ollama is unavailable. The disclaimer is robust. Evaluation infrastructure is needed before paper publication.

---

## Phase 12 — Reproducibility Gaps

- `train_xray_multitask.py` calls `set_seed(42)` ✅
- `train_fusion.py` does NOT call `set_seed()` ❌
- `train_clinical.py` — needs verification
- No git commit hash stored in checkpoint metadata ❌
- No PyTorch/CUDA version pinned in checkpoint metadata ❌

---

## Summary of Confirmed Problems

1. **No completed CORAL ordinal image training run exists.** `best_densenet121_ordinal.pt` (κ=0.0946) is an **interrupted/incomplete** artifact from a run that stopped after 1 of 100 epochs. The CORAL implementation and target encoding are **validated correct** — the gap is an unfinished experiment, not a code defect. Any claim that ordinal regression was used for the image branch is unsupported; the working image checkpoint is non-ordinal.
2. **Missing ordinal safety metrics** (MAE, Within-1, Within-2) not computed in `compute_metrics()`.
3. **Missing calibration evaluation** (ECE, Brier score, reliability diagram).
4. **XAI has no quantitative validation** (insertion/deletion, localization).
5. **RAG has no quantitative evaluation** (retrieval metrics, grounding checks).
6. **OARSI definitional co-occurrence** must be prominently disclosed (see Phase 3).
7. **Fusion training lacks reproducibility seed** and EMA.
8. **RandAugment** may include X-ray-inappropriate operations — not ablated.
9. **Multi-task checkpoint filename ignores the ordinal flag.** `train_xray_multitask.py` writes `best_{model}_multitask.pt` regardless of `--ordinal`, so an ordinal multi-task run would silently overwrite the non-ordinal checkpoint.

---

## Open Questions Requiring Further Work

**1. Fairness of the CORAL vs Focal comparison.**
The two arms currently receive *different* treatment on two axes, so a head-to-head
comparison is not yet apples-to-apples:

- **MixUp target handling.** `MixUpDataset` (active when `MIXUP_ALPHA=0.4`) returns
  soft one-hot targets. `FocalLoss._soft_focal_loss()` consumes them natively, but
  `OrdinalLoss.forward()` reduces them with `targets.argmax(dim=1)`, discarding the
  soft label. A blended KL0/KL4 target becomes whichever class wins the argmax. The
  CORAL arm therefore receives strictly less informative augmentation.
- **Class weighting.** The Focal arm uses inverse-frequency weights
  `[0.506, 1.105, 0.762, 1.527, 6.68]` plus label smoothing; the CORAL arm is
  constructed with `alpha=None` and no smoothing. Additionally, `OrdinalLoss` would
  apply a per-grade weight uniformly to all four cumulative tasks, which is
  inappropriate for a cumulative encoding.

**Required before any comparison is published: a MixUp × class-weight ablation
across both arms.** Until then, no performance claim about CORAL relative to Focal
is supportable in either direction.

**2. CORAL vs Focal at full training length.**
A controlled 6-epoch run (identical dataset, split, backbone, augmentation,
optimizer, LR schedule, batch size and seed; only the loss/head differing) showed
CORAL learning correctly and ahead on the ordinal metrics tested. **This is
explicitly not a superiority claim** — 6 epochs is a truncated, non-converged
comparison, and the arms may converge differently over the full 100-epoch budget.
Re-run to completion, with multiple seeds, before drawing any conclusion.

**3. Completed ordinal training run required.**
Re-run CORAL image training to completion (with resume checkpoints retained) to
obtain a usable ordinal image checkpoint.
