# IMPROVEMENTS.md — KneeVision++ Research Audit Results

> **Correction (2026-09-28).** The CORAL items below (P1, L1, and the corresponding
> table/paper rows) originally described the CORAL ordinal image model as having
> "failed to train" and its checkpoint as "corrupt". A dedicated diagnostic
> (`reports/DIAGNOSTIC_CORAL_ORDINAL.md`, mirrored in `RESEARCH_AUDIT.md`) established
> that **the CORAL implementation and ordinal target encoding are correct**; the
> available checkpoint is an **interrupted epoch-1 artifact** from a 100-epoch run,
> not a corrupt or divergent model. Those statements have been corrected in place.
> All other items are unchanged.

## Confirmed Problems (What Was Actually Wrong)

### P1 — No Completed CORAL Ordinal Image Training Run [CRITICAL — EXPERIMENTAL GAP, NOT A CODE DEFECT]
**The available CORAL checkpoint is an interrupted epoch-1 artifact from a 100-epoch run. The CORAL implementation and ordinal encoding were validated as correct. A completed CORAL training run is still required.**

Distinguishing the four separate states, which the original entry conflated:

| Aspect | Verified status |
|---|---|
| **Implementation correctness** | ✅ **VALIDATED.** `OrdinalLoss`, thresholding, class ordering, output dim, checkpoint loading and metric calculation all verified against a closed-form reference. |
| **Ordinal target encoding** | ✅ **VALIDATED.** `KL0→[0,0,0,0]`, `KL1→[1,0,0,0]`, `KL2→[1,1,0,0]`, `KL3→[1,1,1,0]`, `KL4→[1,1,1,1]`; inverse decode round-trips all five grades. |
| **The existing experiment** | ⚠️ **INTERRUPTED.** `best_densenet121_ordinal.pt` records `best_kappa=0.0946` and `"epoch": 1`. `logs/compare_models.log` ends at the epoch-1 line and no resume checkpoint exists, so the process was terminated during epochs 2–4. Head weights remain at initialization magnitude. |
| **A usable CORAL result** | ❌ **UNFINISHED.** Because `best_kappa` is a running maximum, the epoch-1 value was frozen to disk and never superseded. There is currently **no completed CORAL image model** in the repository. |

The paper claims "CORAL ordinal regression" for the image branch, but the only working image model checkpoint is `best_densenet121.pt` (standard Focal loss, non-ordinal, val kappa=0.775). The fusion model is ordinal, but it is built on a non-ordinal image encoder.

**Impact**: The paper's methodology claim about ordinal regression for the image branch cannot be substantiated from the current artifacts — because the experiment was never finished, not because the method is unsound. The gap is closed by completing a training run, not by fixing code.

### P2 — Missing Ordinal Safety Metrics [HIGH]
`compute_metrics()` did not compute:
- MAE (mean absolute error)
- Within-1 accuracy
- Within-2 accuracy
- ECE (expected calibration error)
- Brier score

These are essential for a clinically-oriented ordinal grading paper.

### P3 — Missing Reproducibility Seed in Fusion Training [MEDIUM]
`train_fusion.py` had no `set_seed()` call and no `--seed` argument, making fusion training results non-reproducible.

### P4 — OARSI Definitional Co-occurrence Must Be Disclosed [HIGH — Research Design]
The BioClinicalBERT branch encodes OARSI per-compartment radiographic grades (JSN, osteophytes, sclerosis, attrition) from `kxr_sq_bu00.txt`. These are the **defining structural components** of the KL grade. The KL grade comes from the same annotation session. This is not conventional label leakage (the KL string is absent from text), but it IS definitional co-occurrence that explains the high text-model kappa (0.953).

### P5 — Calibration Not Evaluated [MEDIUM]
No reliability diagram, ECE, or Brier score was reported anywhere.

### P6 — XAI No Quantitative Validation [MEDIUM]
Grad-CAM/Score-CAM/LIME outputs are visually plausible but no quantitative sanity checks (insertion/deletion tests, localization metrics) were implemented.

### P7 — RAG No Retrieval/Grounding Evaluation [LOW-MEDIUM]
No Precision@K, recall, or hallucination rate measurement for the RAG pipeline.

---

## Fixed Problems (What Was Changed in Code)

### F1 — Added Ordinal & Calibration Metrics to `evaluation/report.py`
**Files changed**: [`src/kneevision/evaluation/report.py`](file:///home/darshan/Projects/Research/KneeVision/src/kneevision/evaluation/report.py)

**New functions**:
- `compute_ordinal_metrics(labels, preds)` → MAE, within-1, within-2, exact accuracy
- `compute_calibration_metrics(labels, probs)` → ECE, per-class Brier score, macro Brier
- `reliability_diagram(labels, probs)` → Reliability diagram figure
- `ordinal_error_plot(labels, preds)` → Ordinal error distribution bar chart

These are now automatically computed inside `compute_metrics()` when `probs` are provided (backward-compatible).

### F2 — Added Reproducibility Seed to `train_fusion.py`
**Files changed**: [`scripts/train_fusion.py`](file:///home/darshan/Projects/Research/KneeVision/scripts/train_fusion.py)

Added `--seed` argument (default 42) and `set_seed()` call. Import of `get_device()` from helpers rather than inline device logic.

### F3 — Created Data Pairing Validation Script
**New file**: [`scripts/validate_data_pairing.py`](file:///home/darshan/Projects/Research/KneeVision/scripts/validate_data_pairing.py)

Produces explicit statistics on:
- Patient-level leakage between splits
- Multimodal pairing match rate
- OARSI field coverage
- OARSI-KL co-occurrence table
- Label agreement between Kaggle and OAI sources

Also writes `DATA_VALIDATION_REPORT.md` and `LEAKAGE_ANALYSIS.md`.

### F4 — Added Tests for New Metrics
**New file**: [`tests/test_ordinal_metrics.py`](file:///home/darshan/Projects/Research/KneeVision/tests/test_ordinal_metrics.py)

31 tests covering: perfect/imperfect MAE, within-1/2 bounds, ECE properties, Brier score range, backward compatibility of `compute_metrics()`, plot smoke tests.

---

## Experiments Performed (During Audit)

| Experiment | Finding |
|---|---|
| Patient leakage check (all splits) | ✅ Zero leakage — 0 shared patients |
| OAI↔Kaggle patient ID matching | 100% match — datasets share subjects |
| Label agreement (Kaggle folder vs OAI CSV) | 99.4% agreement (51/8,260 borderline cases) |
| OARSI field coverage audit | JSN: 100%, Osteophytes: 54%, Sclerosis/Attrition: 33% |
| OARSI-KL monotonicity check | Near-perfect monotonic gradient (confirms definitional co-occurrence) |
| CORAL implementation + ordinal encoding | ✅ Validated correct — encoding table and inverse decode both verified against reference |
| Ordinal CORAL image training run | ⚠️ Interrupted at epoch 1 of 100; no completed CORAL image checkpoint exists yet |
| Test suite (all existing tests) | 199/199 passed before and after changes |
| New metrics tests | 31/31 passed |

---

## Results (Before vs After)

### Metrics Now Computed for Every Evaluation Run

| Metric | Before | After |
|---|---|---|
| Accuracy | ✅ | ✅ |
| Quadratic κ | ✅ | ✅ |
| Macro F1 | ✅ | ✅ |
| Per-class P/R/F1/AUC | ✅ | ✅ |
| **MAE** | ❌ | ✅ |
| **Within-1 accuracy** | ❌ | ✅ |
| **Within-2 accuracy** | ❌ | ✅ |
| **ECE** | ❌ | ✅ |
| **Brier score (per-class + macro)** | ❌ | ✅ |
| Reliability diagram (plot) | ❌ | ✅ |
| Ordinal error distribution (plot) | ✅ (in generate_analysis_curves.py) | ✅ |

### Data Validation (Confirmed Findings)
| Check | Result |
|---|---|
| Patient leakage train/val | 0 patients (✅ clean) |
| Patient leakage train/test | 0 patients (✅ clean) |
| Patient leakage val/test | 0 patients (✅ clean) |
| Duplicate knee entries | 0 (✅ clean) |
| Multimodal match rate | 100% (8,260/8,260 matched) |
| Label agreement | 99.4% (51 borderline discrepancies) |

---

## Remaining Limitations (Cannot Currently Solve)

### L1 — CORAL Image Model Training Run Must Be Completed
The available CORAL checkpoint (`best_densenet121_ordinal.pt`, kappa=0.0946) is an
**interrupted epoch-1 artifact**, not a corrupt checkpoint. The CORAL implementation
and ordinal encoding were validated as correct — this is an unfinished experiment,
not a code bug. The existing `train_xray.py` / `compare_models.py` /
`train_xray_multitask.py` scripts support `--ordinal` correctly.

**Future required experiment**: run CORAL image training to completion (retaining
resume checkpoints), then re-evaluate. Until then no ordinal image result may be cited.

**Action required**: `uv run python scripts/compare_models.py --ordinal --epochs 100 --models densenet121`
(ensure the process can run uninterrupted; `CHECKPOINT_INTERVAL=5` bounds rework to ≤4 epochs)

**Before any CORAL-vs-Focal comparison is published**, a MixUp × class-weight fairness
ablation is also required: `OrdinalLoss` reduces MixUp soft targets via `argmax` while
`FocalLoss` consumes them natively, and only the Focal arm receives class weights. See
"Open Questions" in `RESEARCH_AUDIT.md`.

### L2 — Definitional Co-occurrence Is a Design Constraint, Not a Bug
The high text-model kappa (0.953) from OARSI fields cannot be "fixed" — it reflects the structure of the KL grading system. The correct response is accurate disclosure in the paper, which is now documented in `LEAKAGE_ANALYSIS.md`.

### L3 — KL1 Recall Remains ~56%
KL1 ambiguity is fundamental (human inter-rater κ for KL1 is also low in the literature). The following approaches were not yet tested:
- Hierarchical binary OA/non-OA screener → severity sub-grader
- Higher resolution (320px input)
- Calibrated thresholds (not argmax)

### L4 — XAI Quantitative Validation Infrastructure Missing
No insertion/deletion test, no localization metric against anatomical annotations. Implementing this requires either clinical expert annotations or a principled perturbation benchmark.

### L5 — RAG Evaluation Infrastructure Missing
Grounding accuracy and hallucination rate cannot be measured without a reference question-answer dataset tied to the guideline corpus.

### L6 — External Validation Dataset Not Available
No second knee OA dataset is currently available in the repository. The infrastructure exists (all evaluation scripts take a configurable data path) but no external test set has been prepared.

---

## Recommended Paper Changes

### Must-Change (Factual Accuracy)

1. **Section 4.1**: State that the available CORAL checkpoint (`best_densenet121_ordinal.pt`, kappa=0.0946) is an **interrupted epoch-1 artifact from a 100-epoch run**, and that the CORAL implementation and ordinal encoding were validated as correct — the ordinal image run is unfinished, not failed. The image branch used in all reported experiments is `best_densenet121.pt` (standard Focal loss, non-ordinal). Only the **fusion head** uses CORAL ordinal regression.

2. **Section 5.1 / Discussion**: Add explicit disclosure about OARSI definitional co-occurrence:
   > "The clinical text model's exceptional kappa (0.953) is primarily explained by the fact that the OAI OARSI per-compartment grades (JSN, osteophytes, sclerosis, attrition) encoded in the clinical text are the same structural components that operationally define the KL grade. While this is not label leakage in the engineering sense (the KL grade string is absent from the text), it constitutes definitional co-occurrence that should be understood when interpreting the text-branch performance. In a deployment scenario where OARSI reads are not pre-computed, text-branch performance would degrade substantially."

3. **Section 5 Results**: Add MAE, Within-1, Within-2 accuracy, ECE, and Brier score to all results tables.

### Should-Change (Scientific Completeness)

4. **Section 5.8 Ordinal Error Distribution**: Update with newly computed ECE and Brier score values once the evaluation scripts are re-run with the updated `compute_metrics()`.

5. **Section 9.2 Limitations**: Add KL1 ↔ KL0 confusion analysis. The primary failure mode (KL1 recall ≈56%) is confused with KL0, not KL2 — this changes the clinical interpretation.

### Nice-to-Have

6. Report results from `best_densenet121_multitask.pt` (kappa=0.799 on val) as the correct ablation entry for "DenseNet + OARSI auxiliary heads" — this is the actual strongest image-only checkpoint.

7. Report Within-1 accuracy alongside kappa in the abstract (95.2% is clinically more interpretable than 0.959 for a general audience).
