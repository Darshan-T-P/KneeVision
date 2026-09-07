import { useState } from "react";
import { predictXray, explainXray } from "../api";
import GradeCard from "../components/GradeCard";
import ProbabilityBars from "../components/ProbabilityBars";

export default function XrayPage() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [explain, setExplain] = useState(null);
  const [method, setMethod] = useState("gradcam");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  function onFileChange(e) {
    const f = e.target.files[0];
    if (!f) return;
    setFile(f);
    setPreviewUrl(URL.createObjectURL(f));
    setPrediction(null);
    setExplain(null);
    setError(null);
  }

  async function runDiagnosis() {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const pred = await predictXray(file);
      setPrediction(pred);
      const exp = await explainXray(file, method);
      setExplain(exp);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <h2>🩻 X-ray Diagnosis</h2>
      <p className="page-caption">Upload a knee X-ray for a KL 0-4 grade prediction and a Grad-CAM / Score-CAM / LIME heatmap.</p>

      <div className="controls-row">
        <input type="file" accept="image/*" onChange={onFileChange} />
        <select value={method} onChange={(e) => setMethod(e.target.value)}>
          <option value="gradcam">Grad-CAM</option>
          <option value="scorecam">Score-CAM</option>
          <option value="lime">LIME</option>
        </select>
        <button onClick={runDiagnosis} disabled={!file || loading}>
          {loading ? "Analyzing..." : "Diagnose"}
        </button>
      </div>

      {error && <div className="error-box">{error}</div>}

      <div className="columns">
        {previewUrl && (
          <div className="col">
            <div className="col-title">Original</div>
            <img className="xray-image" src={previewUrl} alt="Uploaded X-ray" />
          </div>
        )}
        {explain && (
          <div className="col">
            <div className="col-title">{method} heatmap</div>
            <img className="xray-image" src={explain.imageUrl} alt="Explanation heatmap" />
          </div>
        )}
        {prediction && (
          <div className="col">
            <GradeCard title="🤖 Prediction" prediction={prediction} />
            <ProbabilityBars probabilities={prediction.probabilities} />
          </div>
        )}
      </div>
    </div>
  );
}
