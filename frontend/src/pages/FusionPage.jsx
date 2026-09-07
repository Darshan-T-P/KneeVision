import { useState } from "react";
import { predictFusion } from "../api";
import GradeCard from "../components/GradeCard";

export default function FusionPage() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [report, setReport] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  function onFileChange(e) {
    const f = e.target.files[0];
    if (!f) return;
    setFile(f);
    setPreviewUrl(URL.createObjectURL(f));
  }

  async function runFusion() {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      setResult(await predictFusion(file, report));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <h2>🔀 Multimodal Fusion</h2>
      <p className="page-caption">Combine the X-ray CNN and clinical-text branches through the trained fusion head.</p>

      <div className="columns">
        <div className="col">
          <input type="file" accept="image/*" onChange={onFileChange} />
          {previewUrl && <img className="xray-image" src={previewUrl} alt="Uploaded X-ray" />}
        </div>
        <div className="col">
          <textarea
            rows={6}
            placeholder="Optional clinical report / findings..."
            value={report}
            onChange={(e) => setReport(e.target.value)}
          />
        </div>
      </div>

      <div className="controls-row">
        <button onClick={runFusion} disabled={!file || loading}>
          {loading ? "Analyzing..." : "Run fusion"}
        </button>
      </div>

      {error && <div className="error-box">{error}</div>}

      {result && (
        <div className="columns">
          <div className="col">
            <GradeCard title="🩻 Image branch" prediction={result.image} />
          </div>
          <div className="col">
            <GradeCard title="📝 Clinical branch" prediction={result.clinical} />
          </div>
          <div className="col">
            <GradeCard title="🧠 Fused decision" prediction={result.fusion} />
          </div>
        </div>
      )}
    </div>
  );
}
