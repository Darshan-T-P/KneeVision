const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function handle(res) {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || JSON.stringify(body);
    } catch {
      // response wasn't JSON — keep statusText
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res;
}

export async function getModelsStatus() {
  const res = await handle(await fetch(`${BASE_URL}/models`));
  return res.json();
}

export async function predictXray(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await handle(await fetch(`${BASE_URL}/predict/xray`, { method: "POST", body: form }));
  return res.json();
}

export async function explainXray(file, method) {
  const form = new FormData();
  form.append("file", file);
  form.append("method", method);
  const res = await handle(await fetch(`${BASE_URL}/predict/xray/explain`, { method: "POST", body: form }));
  const klGrade = res.headers.get("X-KL-Grade");
  const confidence = res.headers.get("X-Confidence");
  const blob = await res.blob();
  return { imageUrl: URL.createObjectURL(blob), klGrade: Number(klGrade), confidence: Number(confidence) };
}

export async function predictClinical(report) {
  const res = await handle(await fetch(`${BASE_URL}/predict/clinical`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ report }),
  }));
  return res.json();
}

export async function predictFusion(file, report) {
  const form = new FormData();
  form.append("file", file);
  form.append("report", report || "");
  const res = await handle(await fetch(`${BASE_URL}/predict/fusion`, { method: "POST", body: form }));
  return res.json();
}

export async function recommendRehab(klGrade, context) {
  const res = await handle(await fetch(`${BASE_URL}/rehab/recommend`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ kl_grade: klGrade, context: context || "" }),
  }));
  return res.json();
}
