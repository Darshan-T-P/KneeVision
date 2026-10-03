# final_ordinal_soft_mixup_a04_test — Frozen Held-Out TEST Evaluation

## Experiment identity
- Model: DenseNet121 + CORAL ordinal head
- Checkpoint: `/home/darshan/Projects/Research/KneeVision/models/best_densenet121_ordinal_ordinal_soft_mixup_a04.pt`
- Checkpoint SHA-256: `c639bd883715637d578d73898116089283c19c0f8b4e30d9bcd51f8d75939f68`
- Training epoch: 25
- Seed: 42
- Selection criteria: val_qwk 0.8008608514358733 (raw, epoch 25); no test data used for selection
- Selected weights: raw
- Test samples: 1656

## Test results
| Metric | Value |
|---|---|
| QWK | 0.823238 |
| MAE | 0.392512 |
| Within-1 accuracy | 0.969203 |
| Within-2 accuracy | 1.000000 |
| Exact accuracy | 0.638285 |
| Macro precision | 0.686609 |
| Macro recall | 0.598791 |
| Macro F1 | 0.626911 |
| Weighted F1 | 0.645963 |
| Linear weighted kappa | 0.683071 |

## Per-class results (KL0–KL4)
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| KL0 | 0.813312 | 0.784038 | 0.798406 | 639 |
| KL1 | 0.338235 | 0.466216 | 0.392045 | 296 |
| KL2 | 0.608051 | 0.642058 | 0.624592 | 447 |
| KL3 | 0.801653 | 0.434978 | 0.563953 | 223 |
| KL4 | 0.871795 | 0.666667 | 0.755556 | 51 |

## Confusion matrix (rows = true KL, columns = predicted KL)
| | KL0 | KL1 | KL2 | KL3 | KL4 | true |
|---|---|---|---|---|---|
| KL0 | 501 | 120 | 18 | 0 | 0 | 639 |
| KL1 | 99 | 138 | 59 | 0 | 0 | 296 |
| KL2 | 16 | 135 | 287 | 9 | 0 | 447 |
| KL3 | 0 | 15 | 106 | 97 | 5 | 223 |
| KL4 | 0 | 0 | 2 | 15 | 34 | 51 |
| predicted | 616 | 408 | 472 | 121 | 39 | 1656 |

## Detailed error analysis
- **KL2 → KL3 errors**: 9 cases
- **KL3 → KL2 errors**: 106 cases
- **Extreme error KL0 → KL4**: 0 cases
- **Extreme error KL4 → KL0**: 0 cases
- **Percentage exactly correct**: 63.83%
- **Percentage within ±1 grade**: 96.92%
- **Percentage within ±2 grades**: 100.00%

### Absolute ordinal error distribution (|True − Predicted|):
| Error Distance | Count | Percentage |
|---|---|---|
| 0 | 1057 | 63.83% |
| 1 | 548 | 33.09% |
| 2 | 51 | 3.08% |
| 3 | 0 | 0.00% |
| 4 | 0 | 0.00% |

## Test-set integrity
- Test sample count: 1656
- Test class counts: [639, 296, 447, 223, 51]
- Unique patients: 828
- Patient overlap with train: 0 (must be 0), with val: 0 (must be 0)
- No test labels were used to select the checkpoint.
- No test metric was fed back into training.
- No threshold or hyperparameter was changed after seeing test results.
- Checkpoint verified unchanged before and after evaluation.

## Integrity statement
This is a frozen held-out evaluation. The checkpoint `/home/darshan/Projects/Research/KneeVision/models/best_densenet121_ordinal_ordinal_soft_mixup_a04.pt` (epoch 25, seed 42, raw weights, validation selection: val_qwk 0.8008608514358733 (raw, epoch 25); no test data used for selection) was selected exclusively using validation QWK. The test split was held out during training and model selection and used only for this final, single-pass evaluation. Test results were NOT used for model selection or any form of tuning.

## Classification report
```
              precision    recall  f1-score   support

         KL0     0.8133    0.7840    0.7984       639
         KL1     0.3382    0.4662    0.3920       296
         KL2     0.6081    0.6421    0.6246       447
         KL3     0.8017    0.4350    0.5640       223
         KL4     0.8718    0.6667    0.7556        51

    accuracy                         0.6383      1656
   macro avg     0.6866    0.5988    0.6269      1656
weighted avg     0.6732    0.6383    0.6460      1656

```
