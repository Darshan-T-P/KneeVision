# Post-hoc Threshold Calibration Report

**Target class:** KL1 recall  
**QWK floor:** 0.77  
**Source:** `models/final_ordinal_soft_mixup_a04_test.predictions.csv`  

## Baseline (argmax) Metrics

| Metric | Value |
|--------|-------|
| QWK | 0.8266 |
| KL0 Recall | 0.8153 |
| KL1 Recall | 0.3986 |
| KL2 Recall | 0.6756 |
| KL3 Recall | 0.4529 |
| KL4 Recall | 0.7059 |

## Calibrated Thresholds (search set)

| Metric | Value |
|--------|-------|
| QWK | 0.7705 |
| KL0 Recall | 0.4648 |
| KL1 Recall | 0.8108 |
| KL2 Recall | 0.4832 |
| KL3 Recall | 0.4484 |
| KL4 Recall | 0.7059 |

**Chosen thresholds:**

| Class | Threshold |
|-------|-----------|
| KL0 | 0.4895 |
| KL1 | 0.1418 |
| KL2 | 0.5000 |
| KL3 | 0.5000 |
| KL4 | 0.5000 |
