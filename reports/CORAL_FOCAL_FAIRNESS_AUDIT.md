# CORAL vs Focal — Fairness Audit

**Objective:** establish the scientifically fairest experimental design to answer
> Under otherwise controlled training conditions, how does CORAL compare with the baseline loss for KL ordinal classification?

**Status:** analysis only. No code changed, no training run, no artifact touched.
The definitive CORAL experiment and its frozen test evaluation are **untouched**.

---

## 1. Existing Focal configuration

The existing baseline is `models/best_densenet121.pt` (Focal, 5-class softmax).
**Provenance caveat:** this artifact comes from a pre-provenance run
(`densenet121_standard_20260822_173636`, finished 2026-08-22, best val QWK
**0.7751262046810463** at epoch 30, early-stopped). Its MLflow params/tags are
**empty**, and its checkpoint has **no `metadata`** block. Its sampler, MixUp,
scheduler, LR and seed settings are therefore *not recoverable*. It cannot serve
as a controlled comparator to the definitive CORAL run.

As it would run *today* through `scripts/compare_models.py` (default, non-ordinal):

| Component | Focal (current harness) |
|---|---|
| Loss | `FocalLoss(gamma=2.0, alpha=class_weights(train,5), label_smoothing=0.1)` |
| Head | 5 logits, argmax decode |
| MixUp targets | soft (native 2-D path in `FocalLoss._soft_focal_loss`) |
| Class weights | `alpha=[0.5055, 1.1048, 0.7623, 1.5266, 6.6798]` (train counts [2286,1046,1516,757,173]) |
| Minority transform | classes {3, 4} get `minority_transform` (extra torsion vs base) |

Notable: under MixUp the targets are **always 2-D** (one-hot or convex mixture),
so Focal always uses the soft-path — `label_smoothing` is **inert** at train time
(it only applies in the unused 1-D hard branch).

## 2. Existing CORAL configuration

`models/best_densenet121_ordinal.pt` — the definitive, audited,
integrity-PASS run (epoch 23, seed 42, raw weights, val QWK 0.7784596290422672,
test QWK 0.7798, 38 epochs / early-stop via `EarlyStopping(patience=15)`).

| Component | CORAL (current harness, `--ordinal`) |
|---|---|
| Loss | `OrdinalLoss(num_classes=5)` — per-threshold BCE on 4 binary tasks `I(grade > k)`, `alpha=None` |
| Head | 4 ordinal logits, `ordinal_to_class` decode (sigmoid→round→sum) |
| MixUp targets | **hardened** — loss does `targets.argmax(dim=1)` on soft 2-D input |
| Class weights | none |
| Label smoothing | none (provenance records `label_smoothing: null` for ordinal) |

## 3. Full fairness matrix

Both arms run through the **same** `Scripts.compare_models.run()` harness — same
code paths, same seed reset *inside* `run()` per model, same loader Generator,
same worker init. That means the paired-run configs are identical except where
flagged.

| Component | Focal | CORAL | Same? |
|---|---|---|---|
| Backbone | DenseNet121 (ImageNet1k V1) | DenseNet121 (ImageNet1k V1) | **Yes** |
| Dataset split | train/val (patient-disjoint) | train/val (patient-disjoint) | **Yes** |
| Seed | `set_seed(seed)` in `run()` | `set_seed(seed)` in `run()` | **Yes** |
| Batch size | 32 | 32 | **Yes** |
| Sampler | `WeightedRandomSampler(1/count^0.5, len, repl=True)` | same | **Yes** |
| MixUp | `MixUpDataset(alpha=0.4)` + CutMix (50/50, apply 50%) | same object, same alpha | **Yes** |
| MixUp target handling | native soft one-hot (convex combo) | `argmax` → hard single grade | **NO — CONFOUND** |
| Augmentation | `train_transform` (+ `minority_transform` for {3,4}) | same | **Yes** |
| Optimizer | AdamW lr 1e-4 wd 1e-4 | same | **Yes** |
| Learning rate | 1e-4 | 1e-4 | **Yes** |
| Weight decay | 1e-4 | 1e-4 | **Yes** |
| Scheduler | LambdaLR warmup `min(5, epochs//6)` + cosine | same | **Yes** |
| Warmup | 5 epochs (for 100-epoch budget) | same | **Yes** |
| EMA | `EMA(decay=0.995)` | same | **Yes** |
| Epoch budget | 100 | 100 | **Yes** |
| Early stopping | `patience=15, min_delta=1e-4` on `best` | same | **Yes** |
| Model selection | `best = max(val_qwk, ema_qwk)` on val only | same | **Yes** |
| Validation metrics | `compute_metrics` → qwk/mae/within-1/2/acc/macro-F1 | same | **Yes** |
| Loss criterion | Focal, γ=2, α=class-weights, ls=0.1 (inert) | plain threshold-BCE, α=None | **Differs (by design)** |
| Head | 5 logits | 4 ordinal logits | **Differs (required)** |
| Decode | argmax(softmax) | round(σ) sum | **Differs (required)** |
| Class weights in loss | inverse-frequency α | none | **Differs** |
| Label smoothing in loss | 0.1 (inert w/ MixUp) | none | **Differs (inert)** |

**Every difference that can affect the comparison:**

1. **MixUp target handling (the primary confound)** — same mixed images, same
   `λ` draws, but CORAL receives a different supervision signal.
2. **Class weights α** — Focal reweights loss by inverse class frequency (KL4 ≈
   6.7×); CORAL weights every threshold equally. This is part of the Focal
   recipe, but it is a *second* difference on top of the ordinal formulation.
3. **Label smoothing** — configured for Focal; inert under MixUp (soft path),
   active only in a no-MixUp Focal cell.
4. **Head/logit topology + decode** — inherent to the methods, not fixable for a
   "fair" comparison, and orthogonal to training conditions.

Items 2–3 are regularization asymmetries; item 1 is a genuine target-representation
confound. Everything else is controlled (identical harness, seed, stream, schedule).

---

## 4. MixUp target-handling analysis

**How targets reach each loss:** `MixUpDataset.__getitem__` returns either a
pure one-hot (50% no-mix branch, or `alpha<=0`) or a convex combination
`λ·e_{a} + (1-λ)·e_{b}` (MixUp or CutMix), always 2-D.

- **Focal:** `FocalLoss` branches on `targets.ndim == 2` to
  `_soft_focal_loss`, which minimizes `CE(logits, soft_target)` (with focal
  modulation on the soft CE). Soft CE is affine in the target vector, so the
  convex-combo target yields the *expected* CE over the two source images —
  a correct, principled use of soft targets.
- **CORAL:** `OrdinalLoss` runs `targets = targets.argmax(dim=1)`
  (`losses.py:44-45`) **before** building the ordinal threshold targets
  `I(y > k)`. For the one-hot/no-mix branch this is the identity (harmless).
  For a genuinely mixed target it **hardens** to the single dominant grade,
  discarding the mixing information and teaching the model that a blended image
  is *exactly* grade `g`.

**Where it happens:** inside `OrdinalLoss.forward`, `src/kneevision/training/losses.py:44–45`.

**Is the conversion mathematically appropriate? No.**
`λ ~ Beta(0.4, 0.4)` gives ≈44% draws <0.4 and ≈44% >0.6; only ≈11% land in
[0.4, 0.6], so most mixed pairs collapse to "the dominant side" — but both
images' pixels are still present, and the target asserts a fully-correct single
grade that is Bayes-inconsistent with the blended input. On a 50/50 tie,
`argmax` resolves to the lower class index. The correct soft CORAL target for
the same mix distribution is the **expected cumulative-tail vector**
`S(k) = P(Y > k) = Σ_{c>k} p_c`, because `BCE_with_logits(logit, t)` is affine in
`t`, hence `Eₜ[BCE] = BCE(logit, E[t])` — i.e. soft-CORAL == expected CORAL loss
over the two constituents, exactly analogous to Focal's soft path.

**Does it give CORAL a different training signal from Focal? Yes.** Under
identical MixUp, Focal minimizes a soft expected-target CE while CORAL minimizes
threshold-BCE against a hardened, partially-wrong single grade. Any Focal/CORAL
QWK gap currently contains this target-handling effect **in addition to** the
ordinal-formulation effect — the two cannot be separated from the existing runs.

**Is native soft-target CORAL feasible? Yes — mathematically and in code.**
Replace the 2-D branch's `argmax` with a cumulative-tail transform
(`S(k) = 1 − prefix-sum(p)` clamped to `[0,1]`) over the class distribution; for
one-hot inputs it is bit-identical to the current hard branch.

---

## 5. Explanation of the confound

A controlled comparison requires both arms to receive the *same* training signal
from the *same* augmented batch stream. The current CORAL arm satisfies the
"same stream" condition but **not** the "same signal": `argmax` silently converts
MixUp's two-class distribution into a single hard label, whereas Focal consumes
the distribution. Because model *selection* (val QWK) and *evaluation* (test QWK)
also depend on the trained weights, this target asymmetry propagates through the
entire pipeline and is indistinguishable, in the final QWK, from a genuine
advantage of the ordinal loss. The existing frozen CORAL numbers therefore
cannot be used to claim a CORAL benefit over the baseline — the two runs never
received the same supervision despite running through the same harness.

## 6. Candidate experimental designs

- **Option A — native soft-target CORAL.** Fix `OrdinalLoss` to accept soft
  targets as the expected cumulative-tail vector. Keeps production MixUp on both
  arms; removes the target-handling confound while preserving the ordinal
  formulation. **Recommended as the headline condition.**
- **Option B — disable MixUp for both.** Targets become scalar ints; the
  one-hot→hard conversion is then the *identity*, so the confound disappears
  entirely. Cleanest possible isolation of the *loss formulation*, at the cost of
  dropping a production training technique from both arms. **Recommended as the
  purity control.**
- **Option C — 2×2 factorial diagnostic.** Adds the four corner comparison
  (Focal±MixUp, CORAL±MixUp). Running all four is **not** required for the
  headline, but the CORAL-hard/MixUp-on corner is *already trained* (the frozen
  definitive run), so this corner is free and lets us measure the confound's
  size directly against a soft-CORAL/MixUp-on rerun. A 2×2 with the existing
  frozen cell reused is scientifically justified and cheap.

## 7. Recommended design

A **paired re-run of both arms through the current harness with identical
conditions**, plus the soft-target fix, reusing the frozen run as the free
confound-diagnostic corner:

| Cell | Arm | MixUp | Loss extras | Purpose |
|---|---|---|---|---|
| H1 | Focal (prod config) | on | α=class-w, ls=0.1 | production parity baseline |
| H2 | CORAL **soft-target** fix | on | — | production parity, corrected signal |
| L1 | Focal minimal | **off** | α=None, ls=0 | pure-loss control |
| L2 | CORAL (soft=identity) | **off** | — | pure-loss control |
| *D* | *frozen CORAL hard-target* | *on* | — | *free diagnostic: size of the `argmax` confound = D − H2* |

Headline: **H2 vs H1** (does corrected CORAL change the outcome under identical
production conditions?). Loss-formulation control: **L2 vs L1** (MixUp off, all
asymmetric regularization stripped — this is the cleanest causal pair). No test
data is used at any decision point.

## 8. Exact configuration for the paired experiment

All via current harness, **new additive CLI flags** (defaults unchanged):

```bash
# H1  Focal, production (MixUp on)
uv run python scripts/compare_models.py --models densenet121 --epochs 100 \
  --batch-size 32 --patience 15 --seed 42 --num-workers 2 --tag focal_mixup

# H2  CORAL, soft ordinal targets (MixUp on)  [requires --soft-ordinal-targets]
uv run python scripts/compare_models.py --models densenet121 --epochs 100 \
  --batch-size 32 --patience 15 --seed 42 --num-workers 2 --ordinal \
  --soft-ordinal-targets --tag ordinal_soft_mixup

# L1  Focal minimal, MixUp off, no class weights / label smoothing
uv run python scripts/compare_models.py --models densenet121 --epochs 100 \
  --batch-size 32 --patience 15 --seed 42 --num-workers 2 --mixup-alpha 0 \
  --class-weights 0 --label-smoothing 0 --tag focal_nomixup

# L2  CORAL, MixUp off
uv run python scripts/compare_models.py --models densenet121 --epochs 100 \
  --batch-size 32 --patience 15 --seed 42 --num-workers 2 --ordinal \
  --mixup-alpha 0 --tag ordinal_soft_nomixup
```

Same split, same seed (per-model reseed in `run()`), same sampler, same
augmentation, same optimizer/scheduler/EMA/early-stop/selection — i.e. the
identical batch stream and identical selection protocol on both arms. The
existing definitive run (CORAL hard, MixUp on) is *reused as-is* for cell **D**;
nothing is retrained for it.

## 9. Required tests before execution

Gaps found: `test_losses.py` covers Focal soft targets but **no test exercises
`OrdinalLoss` with 2-D targets** (the `argmax` path is unprotected), and no test
covers MixUp-flags or artifact isolation. Required additions:

1. `test_ordinal_loss_soft_onehot_equals_hard` — soft(one-hot) ≡ hard for all 5
   grades (identity/back-compat of the 2-D branch).
2. `test_ordinal_loss_soft_expected_value_linearity` — for `p = λe_a + (1-λ)e_b`,
   soft-CORAL loss ≡ `λ·BCE(logit, t_a) + (1-λ)·BCE(logit, t_b)` (the exactness
   property that makes the formulation valid).
3. `test_ordinal_loss_soft_differs_from_hard_on_mixed` — canary proving the
   confound is real (loss differs on a genuinely mixed target).
4. `test_mixup_alpha_zero_returns_onehot` exists — extend to assert the
   no-MixUp arm passes **1-D int labels** into the loss (so soft/hard identity
   holds trivially).
5. `test_compare_models_tag_renames_artifacts_and_checkpoints` — `--tag x` must
   write `best_{model}{suffix}_x.{pt,json,cm.json}` / `checkpoint_{...}_x.pt`
   and **not** open the vanilla filenames for writing.
6. `test_existing_ordinal_artifacts_unchanged_by_tagged_run` — byte-level guard
   that a tagged run leaves `best_densenet121_ordinal.{pt,json,cm.json,test.json}`
   and `checkpoint_densenet121_ordinal.pt` untouched.
7. `test_focal_soft_targets_and_coral_soft_targets_receive_same_mix` (desired,
   if feasible): same seed ⇒ identical λ draws for standard vs ordinal arms,
   guaranteeing the pair is truly the same augmented stream.

Existing protections (unchanged, must stay green): selection rule
`best = max(val_qwk, ema_qwk)` with weight-identity assertion, val-only selection
(`test_selection_split_is_val_not_test`, `test_compare_models_never_loads_test_split`),
resume determinism (`test_resume_continues_from_recorded_epoch`,
`test_resume_matches_fresh_run_bitwise`), MixUp softness (`test_mixup_label_sums_to_one`),
authoritative metrics (`test_ordinal_metrics.py`).

## 10. Expected output / checkpoint naming

All new artifacts are distinct from the frozen set:

| Cell | Checkpoint | Best sidecars |
|---|---|---|
| H1 | `checkpoint_densenet121_focal_mixup.pt` | `best_densenet121_focal_mixup.{pt,json,cm.json}` |
| H2 | `checkpoint_densenet121_ordinal_soft_mixup.pt` | `best_densenet121_ordinal_soft_mixup.{pt,json,cm.json}` |
| L1 | `checkpoint_densenet121_focal_nomixup.pt` | `best_densenet121_focal_nomixup.{pt,json,cm.json}` |
| L2 | `checkpoint_densenet121_ordinal_soft_nomixup.pt` | `best_densenet121_ordinal_soft_nomixup.{pt,json,cm.json}` |
| D (reused, frozen) | `checkpoint_densenet121_ordinal.pt` | `best_densenet121_ordinal.{pt,json,cm.json,test.json}` + `reports/CORAL_DENSENET121_TEST_EVALUATION.md` |

Same seed ⇒ the four new runs share the identical augmented batch stream within
each MixUp condition. Test evaluation, when the time comes, uses the same
frozen test protocol and the existing `evaluate_ordinal_test.py` pattern with
distinct `*_<cell>.test.json` outputs.

## 11. Frozen artifacts statement

The definitive CORAL experiment and its test evaluation **remain frozen and
untouched**:

- `models/best_densenet121_ordinal.pt` (epoch 23, raw, val QWK 0.7784596290422672)
- `models/best_densenet121_ordinal.json`
- `models/best_densenet121_ordinal.cm.json`
- `models/best_densenet121_ordinal.test.json` (test QWK 0.779781, 1656 samples)
- `reports/CORAL_DENSENET121_TEST_EVALUATION.md`
- MLflow run `fed71443b101474980ebd3686d7327ce`, interrupted run `bbf8b86e…` (preserved)

All new runs use separate tag-based checkpoint names; none overwrite the above.

---

## Summary for the causal question

1. **Identified confound(s):** (a) **primary** — MixUp soft labels are hardened
   via `argmax` for CORAL (`losses.py:44-45`) while Focal natively uses the soft
   convex target; (b) **secondary** — Focal additionally reweights by class
   frequency (α) and configures label smoothing (inert under MixUp); both are
   asymmetric regularization on top of the loss formulation; (c) the *existing*
   Focal baseline is unprovenanced (Aug-22) and cannot be compared as-is.
2. **Native soft-target CORAL:** mathematically valid and implementation-feasible
   — `E[BCE] = BCE(E[t])` holds by affinity, so the expected cumulative-tail
   vector `S(k)=P(Y>k)` is the correct 2-D target; one-hot inputs reduce exactly
   to the current hard loss.
3. **Disabling MixUp for both:** yes — it is the **cleaner isolated control** for
   the pure loss-formulation effect (conversion becomes identity), but it drops a
   production technique; the MixUp-on condition is kept for the headline parity
   pair, so both levels are needed.
4. **Next experiment to recommend:** the paired design of §7–§8 — four new runs
   (H1/H2/L1/L2) with `--tag`, `--mixup-alpha`, `--soft-ordinal-targets`,
   `--class-weights 0`, `--label-smoothing 0`, identical seed 42 configuration,
   reusing the frozen definitive CORAL run as the free confound-diagnostic cell.
   H2 vs H1 is the headline; L2 vs L1 is the pure-loss control; D−H2 sizes the
   `argmax` confound.
5. **Code changes required before training:** yes —
   (a) `OrdinalLoss.forward` soft-target branch (`cumulative-tail` instead of
   `argmax`), exposed via opt-in `--soft-ordinal-targets` (default unchanged);
   (b) additive harness flags `--mixup-alpha`, `--class-weights`,
   `--label-smoothing`, `--tag`; (c) the new tests of §9. None of these alter
   defaults, so existing runs/tests remain valid.

No superiority/inferiority claim is made, no winner is declared, and neither
existing experiment was modified to move any number.

**FAIRNESS AUDIT: COMPLETE**