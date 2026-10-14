# Draft manuscript (working text)

Working title: **Ordinal grading of knee osteoarthritis from radiographs with a compact network, and how it behaves at a new hospital**

> This is a DRAFT for the author to edit. Numbers come from result files in this repository (`reports/PAPER_TABLES.md`, `reports/PAPER_TABLES_EXTERNAL.md`, `reports/SEED_ROBUSTNESS.md`, `reports/COMPLEXITY.md`, `reports/external_adapt*/`). Items marked **[VERIFY]** are literature references or facts written from memory that must be checked against the source before use; no reference below has been checked. Do not add claims beyond what the tables support.

---

## Abstract

Knee osteoarthritis (OA) severity is graded from radiographs on the five-level Kellgren–Lawrence (KL) scale, whose neighbouring grades are hard to separate even for experts. We study five-grade KL grading with a compact DenseNet121 (8.0 M parameters) trained with a cumulative ordinal loss using soft targets under MixUp, optionally with auxiliary heads supervised by per-compartment OARSI sub-grades. On a patient-disjoint test set of 1,656 radiographs (OAI-derived), the plain ordinal model reached a quadratic weighted kappa (QWK) of 0.816 ± 0.013 over four seeds, and the model with auxiliary heads 0.828 ± 0.002 over three seeds; about 96% of predictions were within one grade. We then evaluated, without retraining, on an independent hospital cohort of 1,650 radiographs. Performance fell substantially (zero-shot QWK 0.61; binary accuracy 65%, with a tendency to over-call disease). Two inexpensive adaptations recovered much of the gap: re-estimating batch-normalisation statistics on unlabelled target images (QWK 0.68) and re-fitting the ordinal thresholds on 50 labelled target images (QWK 0.72). Training with domain-robust augmentation and per-image standardisation improved zero-shot external QWK from 0.64 to 0.72 (two seeds, matched schedules), and to 0.77 with 50-label recalibration. A simple histogram-equalisation preprocessing worsened external performance. We also report calibration, patient-level bootstrap confidence intervals, and quantitative faithfulness of three explanation methods. The results characterise how much of the apparent accuracy of KL grading survives a change of site, and which low-cost steps help.

---

## 1. Introduction

Knee osteoarthritis is a common degenerative joint disease, and radiographic severity is usually reported with the Kellgren–Lawrence (KL) grade, from 0 (normal) to 4 (severe) **[VERIFY: Kellgren & Lawrence, 1957]**. Grading is subjective: adjacent grades, notably KL1 versus KL0 and KL2, and KL2 versus KL3, are inconsistently assigned between readers, which limits both clinical use and the ceiling of any learned model.

Deep convolutional networks have been applied to automatic KL grading **[VERIFY: e.g. Tiulpin et al., Sci Rep 2018]**. Two issues recur. First, many studies simplify the task, for example by merging grades into a binary diagnosis or by discarding doubtful and minimal cases, which avoids the hardest and most clinically ambiguous distinctions. Second, results are mostly reported on a single cohort without confidence intervals, repeated seeds, or tests at an independent site, so it is unclear how much of the reported accuracy would survive deployment elsewhere.

A recent study by Azad and Anik **[VERIFY: IEEE Access 2025, DOI 10.1109/ACCESS.2025.3631513]** illustrates a strong version of the first approach: a stacked ensemble of three large networks with an image-preprocessing pipeline, reporting 86.29% accuracy for binary diagnosis and 96.93% for a three-class severity task restricted to grades 0, 3 and 4, with an independent-cohort test on a public dataset. We use the same Kaggle/OAI data and split, and the same independent cohort, but address the full five-grade problem with a single compact network, and we emphasise measurement of uncertainty and of behaviour under site shift.

**Contributions.**
1. A five-grade ordinal KL grader based on a single 8.0 M-parameter DenseNet121 (about ten times fewer parameters than the compared ensemble), trained with soft cumulative targets under MixUp, with an optional multitask variant using OARSI sub-grade supervision; results over multiple seeds with patient-level bootstrap confidence intervals and calibration.
2. A measured external-validation study on an independent hospital cohort showing a substantial zero-shot drop, and a systematic account of what recovers it: unsupervised batch-normalisation re-estimation, few-label threshold recalibration, and domain-robust training. We also report a negative result for histogram equalisation.
3. A quantitative faithfulness evaluation (insertion/deletion) of Grad-CAM, Score-CAM and LIME on this task.

---

## 2. Related work (to be completed by the author)

Keep this short and cited. Suggested subsections, each to be filled from verified sources: (a) CNN-based KL grading on OAI; (b) ordinal regression for medical grading, including cumulative binary formulations **[VERIFY: Frank & Hall 2001; Cao et al., CORAL, Pattern Recognit Lett 2020]**; (c) MixUp and its interaction with ordinal labels **[VERIFY: Zhang et al., ICLR 2018]**; (d) domain shift and adaptation in medical imaging, including adaptive batch normalisation **[VERIFY: Li et al., Pattern Recognition 2018]**; (e) explainability for radiographs **[VERIFY: Selvaraju et al. 2017; Wang et al. 2020; Ribeiro et al. 2016]**.

---

## 3. Methods

### 3.1 Data

*Primary cohort.* We used a publicly available Kaggle knee-radiograph dataset derived from the Osteoarthritis Initiative (OAI) **[VERIFY: dataset citation and licence]**, with KL grades 0–4. It contains 8,260 images split by the dataset into training (5,778), validation (826) and test (1,656). Patient identifiers are encoded in file names (OAI participant ID plus side). We verified that no patient appears in more than one split (training 2,889, validation 413, test 828 patients; zero overlap). Class counts in the test set were 639 / 296 / 447 / 223 / 51 for KL0–KL4. Images were resized to 224 × 224.

*Independent cohort.* The Gornale and Patravali "Digital Knee X-ray Images" dataset **[VERIFY: Mendeley Data, doi:10.17632/t9ndx37v5h.1, CC BY 4.0]** contains 1,650 radiographs collected from hospitals and diagnostic centres on a different X-ray system and labelled with KL grades by two experts. We used Expert I labels (counts 514 / 477 / 232 / 221 / 206 for KL0–KL4); Expert II labels differ for 11 images. Images are 300 × 162 pixel RGB PNG files and were resized to 224 × 224. The cohort has no patient identifiers. We confirmed that two other public exports we initially considered (a Roboflow export and a tibial-spiking crop set) are re-distributions of OAI images and therefore are not independent.

### 3.2 Model

The backbone is an ImageNet-pretrained DenseNet121 followed by a head of dropout (0.5), a 1,024-unit linear layer with batch normalisation and ReLU, dropout (0.3), and a final linear layer. For ordinal models the final layer has four outputs, one per threshold "grade > k" (k = 0…3), each with its own weights. This is a cumulative-binary ("CORAL-style") formulation trained with the CORAL cumulative loss, but it does **not** share weights across thresholds as in the original CORAL, so rank consistency is not guaranteed by construction. The predicted grade is the number of thresholds whose sigmoid is at least 0.5; class probabilities for AUC and calibration are derived from the cumulative probabilities. The optional multitask variant adds auxiliary linear heads predicting the per-compartment OARSI sub-grades available for the same knees in the OAI clinical tables (joint-space narrowing, osteophytes, sclerosis and attrition, medial and lateral; four levels each, missing values ignored), with loss weight 0.3; auxiliary heads are not used at inference.

### 3.3 Ordinal loss with soft targets under MixUp

With MixUp or CutMix, a training label is a distribution p over the five grades. We use the cumulative targets t_k = P(Y > k) = Σ_{c>k} p_c and minimise the mean binary cross-entropy over the four thresholds. Because binary cross-entropy is affine in its target, this equals the expected hard-label loss under the mixing distribution, and for one-hot labels it reduces to the standard cumulative loss. The baseline for comparison hardens mixed labels with an argmax before building thresholds.

### 3.4 Training

AdamW (learning rate 1e-4, weight decay 1e-4), batch size 32, gradient-norm clipping at 1.0, mixed precision, 5-epoch linear warm-up followed by cosine decay over a 100-epoch budget, exponential moving average of weights (decay 0.995), and early stopping with patience 15 on validation QWK; the checkpoint with the highest validation QWK (raw or EMA weights) is kept. Sampling used a class-frequency-weighted sampler (inverse frequency to the power 0.5). Augmentation: random resized crop, horizontal flip, rotation, colour jitter, RandAugment, affine translation, sharpness, blur and random erasing; MixUp/CutMix with α = 0.4 applied to half of the samples. Seeds 42, 7, 123 and 2024 were used for the plain ordinal model and 42, 7 and 2024 for the multitask model. All model selection used the validation split only; the test split was evaluated once per final checkpoint.

### 3.5 Baselines and ablations

We compared against a multi-class Focal-loss model (3 seeds), against variants without MixUp or with hard targets, and against ConvNeXt-Tiny, EfficientNet-B4 and Swin-T trained with the same recipe (one seed each; the recipe was tuned on DenseNet121). A preprocessing ablation applied global histogram equalisation at training and test time.

### 3.6 Evaluation protocol

*Internal.* QWK (primary), MAE, within-1 and within-2 accuracy, exact accuracy, macro-F1, per-class precision/recall/F1 and one-vs-rest AUC, derived binary (KL0–1 vs KL2–4) and three-group (KL0–1 / KL2 / KL3–4) results, and calibration: expected calibration error (top-label and class-wise) and Brier score. Confidence intervals use a patient-level cluster bootstrap (1,000 resamples of the 828 test patients, because a patient's two knees are not independent). Multi-seed results are reported as mean ± sample standard deviation.

*External.* No external image was used for training or for any model-selection decision. For the adaptation study we split the external cohort once (stratified by grade, fixed seed) into a 30% development part (495 images) and a 70% held-out part (1,155 images). Every adaptation used only the development part, and all reported external numbers are on the held-out part. Methods: (i) zero-shot; (ii) AdaBN: batch-normalisation statistics re-estimated on unlabelled development images (a full-network adaptation only for DenseNet121 and EfficientNet-B4; for ConvNeXt-Tiny and Swin-T only the single batch-normalisation layer in the head is affected); (iii) threshold recalibration: the four threshold biases are fitted to maximise QWK on 50, 100 or all 495 labelled development images (random draws averaged where fewer than all); (iv) AdaBN plus recalibration. The external cohort lacks patient identifiers, so the same patient may appear in both parts, which could make adapted results slightly optimistic.

### 3.7 Domain-robust training

Specified before the experiment: random aspect-ratio change, random down- and up-sampling, random gamma, stronger brightness and contrast jitter, Gaussian blur and noise, and per-image standardisation (zero mean, unit variance per image) at training and test time. Because we shortened the schedule for time, it was compared only with a matched baseline trained with the same 40-epoch schedule and patience of 8 (seeds 42 and 7), and these results are not compared with the 100-epoch results.

### 3.8 Explanation faithfulness

On validation images, saliency maps from Grad-CAM, Score-CAM and LIME were converted to a ranking of 16 × 16 patches. We tracked the predicted-grade probability while revealing (insertion) or removing (deletion) patches in decreasing saliency order and report the area under each curve, against a random-saliency baseline.

### 3.9 Complexity

Parameters, checkpoint size and FLOPs (2 × multiply-accumulates for one 224 × 224 image, main head only) were measured, and latency was measured on a single laptop GPU; latency is hardware-specific.

---

## 4. Results

*(Use `docs`/`reports` tables for exact figures; text below is the narrative.)*

### 4.1 Internal grading
Over four seeds the plain ordinal model reached a test QWK of 0.816 ± 0.013, MAE 0.408 ± 0.018, and within-1 accuracy 96.1 ± 1.1%. The multitask model reached 0.828 ± 0.002, MAE 0.389 ± 0.005 and within-1 95.8 ± 0.4% over three seeds, an improvement of about 0.011 QWK with much lower seed-to-seed variation. Exact accuracy was 63.2 ± 1.9% and 65.4 ± 1.0%. The Focal baseline (no MixUp, three seeds) reached QWK 0.785 ± 0.023. Most errors were between adjacent grades; per-class recall (multitask) was 83%, 33%, 69%, 44% and 93% for KL0–KL4, so KL1 and KL3 were the most difficult. Class-wise ECE was 0.038–0.045. Backbones trained with the same recipe reached QWK 0.805 (ConvNeXt-Tiny), 0.782 (EfficientNet-B4) and 0.817 (Swin-T) with 2–4 times more parameters than DenseNet121.

### 4.2 Comparison with the base study's tasks
On the binary task (KL0–1 vs KL2–4), derived from our five-grade predictions, accuracy was 85.1 ± 1.0% (plain) and 86.0 ± 0.6% (multitask), compared with 86.29% reported for the ensemble. On the three-grade task restricted to KL0, KL3 and KL4, our model's accuracy was about 95% when its choice was limited to those grades by probability (an approximation to their task), against 96.93% reported. The inference model is 8.0 M parameters, 33 MB and 5.67 GFLOPs, versus 77.3 M parameters, 335 MB and 13.14 GFLOPs reported for the ensemble.

### 4.3 External validation
Zero-shot on the held-out external images, the plain ordinal model reached QWK 0.610 ± 0.034 (four seeds), binary accuracy 65.0 ± 3.0%, sensitivity 96.4% and specificity 44.1%: grades were over-called. AdaBN raised QWK to 0.682 ± 0.031 (binary accuracy 78.4%, specificity 74.4%). Recalibration with 50 labelled images gave QWK 0.715 ± 0.029 and with all 495 gave 0.733 ± 0.020. Combining AdaBN and recalibration was not better than recalibration alone for DenseNet121. Histogram equalisation reduced external QWK (0.35 vs 0.56 for the same seed on the full cohort). With domain-robust training, matched-baseline versus robust external zero-shot QWK was 0.638 ± 0.048 versus 0.720 ± 0.035, and after 50-label recalibration 0.715 ± 0.013 versus 0.773 ± 0.001; internal test QWK was 0.794 ± 0.026 versus 0.827 ± 0.010, with within-1 accuracy 96.1% versus 94.1%. External exact five-grade accuracy stayed below 55% in all settings.

### 4.4 Explanations
Insertion AUC was 0.357 (Grad-CAM, n = 100), 0.362 (Score-CAM, n = 20) and 0.293 (LIME, n = 30) versus 0.237 for random saliency. Deletion AUC was 0.217, 0.248 and 0.259 versus 0.238 (random), so only Grad-CAM was (marginally) better than chance on deletion.

---

## 5. Discussion

**Internal accuracy and what it does and does not show.** Single-network ordinal grading reaches QWK above 0.8 with about 96% of predictions within one grade, and exact accuracy in the mid-60s, which is consistent with the known ambiguity between adjacent KL grades. The soft-target formulation avoids discarding mixed-label information, and multitask supervision slightly improves agreement and stabilises seeds. Because the auxiliary labels are the radiographic components that define KL, the benefit should be read as better supervision of the same images, not as independent clinical information; for the same reason the high scores of the text and fusion branches in our larger project are not evidence of independent signal and are not claimed here.

**Comparison with the base study.** On the binary task we are level with the reported ensemble using a model about ten times smaller, and slightly lower on its restricted three-grade task. We do not claim superiority on those tasks. The contribution is the full five-grade problem with uncertainty estimates, and the measurement of site shift.

**Site shift.** The principal weakness is external. Without any adaptation, agreement fell from about 0.82 to about 0.61, mostly through over-calling disease (sensitivity 96%, specificity 44%) while the ranking of cases stayed informative (binary AUC about 0.83 on the held-out part). The pattern, large gains from re-estimating batch-normalisation statistics and from re-fitting only four threshold offsets, suggests that a substantial part of the loss comes from intensity and decision-threshold shift rather than from lost discriminative features. We have not isolated causes, and part of the gap may reflect different annotators and image formats (the external images are small, wide crops that we resized to a square). Training with domain-robust augmentation improved both external and internal agreement in both seeds, which suggests that the standard augmentation was too narrow, but this rests on two seeds and a shortened schedule.

**What did not work.** Global histogram equalisation, a component of the base study's preprocessing, lowered external performance in our setting. It does not follow that the base study's full pipeline or ensemble is ineffective; we did not reproduce it, and our result concerns this one preprocessing step in a single-network setting.

**Practical implication.** If a model like this were deployed at a new site, our results indicate that a modest local calibration step (for example, 50 labelled images, or unlabelled images for batch-normalisation statistics) should be part of deployment, and that performance should be re-measured there. Adapted results use target-site data and must not be quoted as zero-shot generalisation.

**Explanations.** The three methods gave attributions that were above chance on insertion but not clearly better than chance on deletion, with large per-image variability; they should be treated as indicative visualisations rather than precise localisation.

## 6. Limitations

1. One training cohort; independent-site performance is substantially lower, and our adaptations reduce but do not remove the gap. Zero-shot external binary accuracy (best case about 74%) remains below the 85.7% reported by the base study on this cohort.
2. External adaptation results use target-site data; the cohort has no patient IDs, so the development/test split may share patients.
3. External labels come from different annotators; KL1 is particularly inconsistent between sources.
4. KL1 and KL3 recall are low and five-grade exact accuracy is about 63–66% internally and below 55% externally.
5. Auxiliary OARSI supervision uses the components that define KL.
6. Small numbers of seeds (2–4); differences below about 0.01–0.03 QWK should not be read as significant. Backbone comparisons used one seed and a recipe tuned on DenseNet121. The 40-epoch robust-training study has two seeds and is not comparable with the 100-epoch results.
7. The ordinal head is cumulative-binary with per-threshold weights, so rank consistency is not guaranteed.
8. Explanation faithfulness was measured on small subsets and deletion results were close to chance.
9. Retrospective research prototype; not a validated diagnostic device and not evaluated prospectively or against clinician readers.

## 7. Conclusion

A compact single network with a cumulative ordinal loss and soft MixUp targets grades all five KL levels with QWK about 0.82–0.83 on a patient-disjoint OAI-derived test set, using roughly one tenth of the parameters of a published ensemble. On an independent hospital cohort performance fell markedly, but unsupervised or few-label adaptation and domain-robust training recovered a large share of the loss. Reporting site-shift behaviour, repeated seeds and calibration alongside accuracy gives a more realistic picture of what such systems deliver. Future work: a second independent cohort, more seeds for domain-robust training with the standard 100-epoch schedule, and prospective comparison with clinician readers.

---

## References to collect and verify

Kellgren & Lawrence 1957; the OAI; the Kaggle dataset citation; Azad & Anik 2025 (IEEE Access, DOI as printed in the PDF in this repository); Gornale & Patravali (Mendeley Data, DOI above); Tiulpin et al. 2018; Frank & Hall 2001; Cao, Mirjalili & Raschka 2020 (CORAL); Zhang et al. 2018 (MixUp); Huang et al. 2017 (DenseNet); Li et al. 2018 (AdaBN); Selvaraju et al. 2017 (Grad-CAM); Wang et al. 2020 (Score-CAM); Ribeiro et al. 2016 (LIME). None of these has been checked; confirm authors, venue, year and DOI before use.
