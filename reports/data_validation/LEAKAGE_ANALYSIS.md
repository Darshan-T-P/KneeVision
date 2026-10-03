# LEAKAGE_ANALYSIS.md

## Summary

**Engineering leakage** (label string in features): ❌ NOT present.
`compose_clinical_report()` explicitly excludes the KL grade field.

**Definitional co-occurrence** (OARSI components define KL): ⚠️ PRESENT.
This is the fundamental experimental design issue.

## What Is Included in Clinical Text

- Medial/lateral JSN grade (0–3 ordinal, from kxr_sq_bu00.txt)
- Medial/lateral osteophyte grade (0–3)
- Medial/lateral subchondral sclerosis (0–3)
- Medial/lateral bone attrition (0–3)
- Patient demographics (age, BMI)
- WOMAC symptom scores (pain, stiffness, function) — 0-100
- Injury/surgery/medication history

## Why This Is Not Conventional Label Leakage

1. The KL grade **string** does not appear in the text.
2. The OARSI sub-component grades and the KL grade come from the same
   radiologist reading session but are **separate annotation outputs**.
3. A radiologist seeing only the sub-component grades could deterministically
   reconstruct the KL grade — but this is also true in clinical practice.
4. This mirrors how the KL system works in reality.

## Why This IS an Experimental Design Problem

1. The BioClinicalBERT model achieves κ=0.953 primarily because it learns
   to decode the near-deterministic relationship between OARSI components
   and KL grade — not because it learns generalizable disease severity.
2. In a real-world deployment scenario, OARSI sub-component reads may not
   be available as input — they require the same radiologist reading that
   would produce the KL grade itself.
3. The high text-model performance should not be interpreted as
   'clinical text adds independent information beyond imaging.'

## Required Paper Disclosures

The paper MUST clearly state:
- The clinical text features are derived from the SAME OAI annotation
  session (kxr_sq_bu00.txt) that also produces the KL grade label.
- OARSI sub-components are the structural defining components of KL grades.
- The text model's high performance reflects this definitional relationship.
- In a deployment scenario without pre-existing OARSI reads, the text
  branch's performance would degrade substantially.

## OARSI-KL Monotonic Evidence

| KL Grade | Mean JSN_M | Mean Osteophyte_M |
|---|---|---|
| KL0 | 0.000 | 0.019 |
| KL1 | 0.398 | 0.387 |
| KL2 | 0.529 | 1.127 |
| KL3 | 1.626 | 1.762 |
| KL4 | 2.051 | 2.538 |

The near-perfect monotonic relationship confirms definitional co-occurrence.