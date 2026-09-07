# OAI Clinical Dataset — Access & Setup

The clinical dataset for KneeVision Phase 4 comes from the **Osteoarthritis
Initiative (OAI)**, hosted in the **NIMH Data Archive (NDA)**.

- **Portal:** https://nda.nih.gov/oai
- **Access:** free, but gated — NDA account + Data Use Certification + approval.
- **Docs:** https://nda.nih.gov/oai/study_documentation

## Why OAI

- 4,796 participants, 9-year follow-up, knee X-rays + clinical outcomes.
- Human expert **KL grades** (OAI Project 15 central readings) per knee —
  the same gold-standard labels used to grade our X-ray images.
- Patient-reported outcomes (WOMAC / KOOS), demographics, risk factors,
  comorbidities — the "clinical" modality for multimodal fusion.
- Note: OAI provides **structured** clinical data, not free-text radiology
  reports. For narrative reports, see the expert-annotated OAI report dataset
  on IEEE DataPort (DOI `10.21227/vcpg-qm58`, subscription required).

## Registration & Approval (one-time, takes days–weeks)

1. Create an NDA account: https://nda.nih.gov/nda/creating-an-nda-account
2. Log in through the **Research Auth Service (RAS)** using an eRA Commons,
   Login.gov, or PIV/CAC account.
3. Accept the OAI **Data Use Certification** (data-access terms) on the OAI
   portal and submit your request.
4. Wait for approval (emailed).

## Which tables to download

Put these in `data/oai/raw/` (the ingest script reads exactly these names):

| File | Contents | Why |
|------|----------|-----|
| `kxr_sq_bu00.txt` | Baseline semi-quantitative X-ray readings (KL grades, osteophytes, JSN, sclerosis, attrition) | Gold-standard KL label, **plus** the real per-compartment findings used to compose the clinical report text |
| `AllClinical00.txt` | Baseline clinical variables (age, sex, BMI, WOMAC, risk factors) | Clinical features for the text model |
| `kxr_sq_bu01/03/05/06/08.txt` | Follow-up readings (optional) | Longitudinal analysis |

Downloads are under the portal's **Download Complete Datasets** page
("X-Ray Image Assessments_ASCII" and "AllClinical"). Each file is a
delimiter-separated ASCII table; no DICOM images are required for the
clinical pipeline.

## Build the dataset (once files are in place)

```bash
# check which files are present + re-print these steps
uv run python scripts/download_oai.py status

# build the clinical dataset
uv run python scripts/download_oai.py ingest

# train the clinical-text model on the result
uv run python scripts/train_clinical.py --data data/oai/processed/oai_clinical.csv --epochs 10
```

Outputs produced by `ingest`:

- `data/oai/processed/oai_clinical.csv` — one row per knee: `id, side, split,
  kl_grade, report, age, sex, bmi, pain, stiffness, function, injury, surgery,
  meds, jsn_m, jsn_l, osteophyte_m, osteophyte_l, sclerosis_m, sclerosis_l,
  attrition_m, attrition_l`. Subjects are split into train / val / test
  **without ever splitting a participant** across splits.
- `data/oai/reports/{train,val,test}/{kl_grade}/report_{id}_{side}.txt` —
  the folder layout accepted by `load_reports_from_folders`.

## How the clinical "reports" are built

OAI has no free-text reports, so each knee record is a **structured clinical
summary** compiled from two kinds of real OAI fields:

1. **Real per-compartment OARSI radiographic grades** — medial/lateral joint
   space narrowing, osteophytes, subchondral sclerosis, and attrition,
   parsed from the same `kxr_sq_bu00.txt` file the KL grade itself comes
   from. These are the independent structural components a radiologist uses
   to arrive at the KL grade, rendered as a "RADIOGRAPHIC FINDINGS:" section
   (see `compose_radiographic_findings` in `src/kneevision/clinical/prepare.py`).
2. **Demographics and patient-reported symptoms** — age, sex, BMI, WOMAC
   pain/stiffness/function, injury/surgery/medication history.

The KL grade itself is the **label** — it is deliberately never in the text.
Including the real radiographic components (as opposed to only demographics)
makes the clinical-text task much easier than "infer severity from symptoms
alone" (see the caveat in the main README's Phase 5 section) — it is closer
to decoding KL's own defining components written as prose, which is still a
legitimate, leakage-free result but a different claim than modeling patient
narrative alone.

## Citation

```
Osteoarthritis Initiative. NIMH Data Archive. https://nda.nih.gov/oai
```

IEEE DataPort reports (if you later want real narrative text):
Akshay Daydar. "Expert annotated radiology Reports for Knee OAI dataset."
IEEE DataPort, 10.21227/vcpg-qm58.