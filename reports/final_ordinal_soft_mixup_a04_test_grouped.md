# Post-hoc clinical regrouping (derived, no new inference)

**Source (frozen, untouched):** `models/final_ordinal_soft_mixup_a04_test.json`
**Source checkpoint SHA-256:** `c639bd883715637d578d73898116089283c19c0f8b4e30d9bcd51f8d75939f68`
**Method:** pure arithmetic re-aggregation of the already-frozen 5x5 test confusion matrix (['KL0', 'KL1', 'KL2', 'KL3', 'KL4']). No model inference, no checkpoint reload, no test-split reload was performed to produce this file.

## binary grouping: ['No OA (KL 0-1)', 'OA (KL 2-4)']

- Accuracy: 85.33%
- Macro F1: 0.8482
- Quadratic-weighted kappa: 0.6973

| Group | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No OA (KL 0-1) | 0.8379 | 0.9176 | 0.8760 | 935 |
| OA (KL 2-4) | 0.8782 | 0.7698 | 0.8204 | 721 |

Confusion matrix (rows=true, cols=pred, order=['No OA (KL 0-1)', 'OA (KL 2-4)']):
```
[858, 77]
[166, 555]
```

## 3class grouping: ['None/Doubtful (KL 0-1)', 'Mild (KL 2)', 'Moderate/Severe (KL 3-4)']

- Accuracy: 78.26%
- Macro F1: 0.7321
- Quadratic-weighted kappa: 0.7622

| Group | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| None/Doubtful (KL 0-1) | 0.8379 | 0.9176 | 0.8760 | 935 |
| Mild (KL 2) | 0.6081 | 0.6421 | 0.6246 | 447 |
| Moderate/Severe (KL 3-4) | 0.9437 | 0.5511 | 0.6959 | 274 |

Confusion matrix (rows=true, cols=pred, order=['None/Doubtful (KL 0-1)', 'Mild (KL 2)', 'Moderate/Severe (KL 3-4)']):
```
[858, 77, 0]
[151, 287, 9]
[15, 108, 151]
```
