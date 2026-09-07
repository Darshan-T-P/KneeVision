import { useState } from "react";
import { predictClinical } from "../api";
import GradeCard from "../components/GradeCard";
import ProbabilityBars from "../components/ProbabilityBars";

const SAMPLE_REPORT =
  "RADIOGRAPHIC FINDINGS: Medial compartment: moderate joint space narrowing, definite osteophytes, " +
  "moderate subchondral sclerosis. Lateral compartment: no joint space narrowing, small osteophytes, " +
  "no subchondral sclerosis.";

export default function ClinicalPage() {
  const [report, setReport] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function runPrediction() {
    if (!report.trim()) return;
    setLoading(true);
    setError(null);
    try {
      setPrediction(await predictClinical(report));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <h2>📝 Clinical Text</h2>
      <p className="page-caption">Paste a radiology report / clinical findings to get a KL 0-4 grade from BioClinicalBERT.</p>

      <textarea
        rows={6}
        placeholder="FINDINGS: ..."
        value={report}
        onChange={(e) => setReport(e.target.value)}
      />
      <div className="controls-row">
        <button onClick={() => setReport(SAMPLE_REPORT)}>Load sample</button>
        <button onClick={runPrediction} disabled={!report.trim() || loading}>
          {loading ? "Analyzing..." : "Predict"}
        </button>
      </div>

      {error && <div className="error-box">{error}</div>}

      {prediction && (
        <div className="columns">
          <div className="col">
            <GradeCard title="🤖 Prediction" prediction={prediction} />
          </div>
          <div className="col">
            <ProbabilityBars probabilities={prediction.probabilities} />
          </div>
        </div>
      )}
    </div>
  );
}
