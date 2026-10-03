# CORAL/DenseNet121 — Frozen Held-Out TEST Evaluation

## Experiment identity
- Model: DenseNet121 + CORAL ordinal head
- Checkpoint: `models/best_densenet121_ordinal.pt`
- Training epoch: 23
- Seed: 42
- Validation QWK used for selection: 0.7784596290422672
- Selected weights: raw
- Test samples: 1656

## Test results
| Metric | Value |
|---|---|
| QWK | 0.779781 |
| MAE | 0.457729 |
| Within-1 accuracy | 0.956522 |
| Within-2 accuracy | 0.999396 |
| Exact accuracy | 0.586353 |
| Macro precision | 0.656206 |
| Macro recall | 0.533312 |
| Macro F1 | 0.556645 |
| Weighted F1 | 0.583511 |
| Linear weighted kappa | 0.622689 |

## Per-class results (KL0–KL4)
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| KL0 | 0.774691 | 0.785603 | 0.780109 | 639 |
| KL1 | 0.309133 | 0.445946 | 0.365145 | 296 |
| KL2 | 0.531120 | 0.572707 | 0.551130 | 447 |
| KL3 | 0.774194 | 0.215247 | 0.336842 | 223 |
| KL4 | 0.891892 | 0.647059 | 0.750000 | 51 |

## Confusion matrix (rows = true KL, columns = predicted KL)
| | KL0 | KL1 | KL2 | KL3 | KL4 | true |
|---|---|---|---|---|---|
| KL0 | 502 | 121 | 16 | 0 | 0 | 639 |
| KL1 | 112 | 132 | 52 | 0 | 0 | 296 |
| KL2 | 33 | 157 | 256 | 1 | 0 | 447 |
| KL3 | 1 | 17 | 153 | 48 | 4 | 223 |
| KL4 | 0 | 0 | 5 | 13 | 33 | 51 |
| predicted | 648 | 427 | 482 | 62 | 37 | 1656 |

## Test-set integrity
- Test sample count: 1656
- Test class counts: [639, 296, 447, 223, 51]
- Unique patients: 828
- Patient overlap with train: 0 (must be 0), with val: 0 (must be 0)
- No test labels were used to select the checkpoint.
- No test metric was fed back into training.
- No threshold or hyperparameter was changed after seeing test results.

## Integrity statement
This is a frozen held-out evaluation. The checkpoint `models/best_densenet121_ordinal.pt` (epoch 23, seed 42, raw weights, validation QWK 0.7784596290422672) was selected exclusively using validation QWK/EMA QWK. The test split was held out during training and model selection and used only for this final, single-pass evaluation. Test results were NOT used for model selection or any form of tuning.

## Classification report
```
              precision    recall  f1-score   support

         KL0     0.7747    0.7856    0.7801       639
         KL1     0.3091    0.4459    0.3651       296
         KL2     0.5311    0.5727    0.5511       447
         KL3     0.7742    0.2152    0.3368       223
         KL4     0.8919    0.6471    0.7500        51

    accuracy                         0.5864      1656
   macro avg     0.6562    0.5333    0.5566      1656
weighted avg     0.6293    0.5864    0.5835      1656

```
