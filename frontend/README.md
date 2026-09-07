# KneeVision++ Frontend

A Vite + React single-page app for the KneeVision++ REST API (Phase 7b) — a decoupled
alternative to the Streamlit demo (`../streamlit_app.py`), talking to the same models
over HTTP instead of loading them in-process.

## Pages

- **🩻 X-ray Diagnosis** — upload a knee X-ray, get a KL 0-4 prediction and a
  Grad-CAM / Score-CAM / LIME heatmap.
- **📝 Clinical Text** — paste a radiology report / findings, get a KL 0-4
  prediction from BioClinicalBERT.
- **🔀 Fusion** — upload an X-ray + optional report, see the image branch,
  clinical branch, and fused decision side by side.
- **🏃 Rehab** — retrieval-augmented rehab guidance for a KL grade (see the
  main README's Phase 6 section).

## Run

Start the API first (from the project root):

```bash
uv run uvicorn kneevision.api.main:app --reload --port 8000 --app-dir src
```

Then, in this directory:

```bash
npm install
npm run dev       # http://localhost:5173
```

The API base URL defaults to `http://localhost:8000`; override it with a
`VITE_API_URL` env var (e.g. in a `.env.local` file) if the API runs elsewhere.

## Other commands

```bash
npm run lint      # oxlint
npm run build     # production build to dist/
npm run preview   # serve the production build locally
```
