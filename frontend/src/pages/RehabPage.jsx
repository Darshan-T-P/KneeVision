import { useState } from "react";
import { recommendRehab } from "../api";

export default function RehabPage() {
  const [klGrade, setKlGrade] = useState(2);
  const [context, setContext] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function runRecommend() {
    setLoading(true);
    setError(null);
    try {
      setResult(await recommendRehab(klGrade, context));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <h2>🏃 Rehab Recommendation</h2>
      <p className="page-caption">
        Retrieval-augmented guidance from public OA rehab guidelines (OARSI, AAOS, ACR/Arthritis Foundation, CDC),
        synthesized by a local LLM. Not medical advice.
      </p>

      <div className="controls-row">
        <label>
          KL grade
          <input
            type="range"
            min={0}
            max={4}
            value={klGrade}
            onChange={(e) => setKlGrade(Number(e.target.value))}
          />
          <strong> KL {klGrade}</strong>
        </label>
        <input
          type="text"
          placeholder="Optional patient context (e.g. 68yo, BMI 31, pain climbing stairs)"
          value={context}
          onChange={(e) => setContext(e.target.value)}
          style={{ flex: 1 }}
        />
        <button onClick={runRecommend} disabled={loading}>
          {loading ? "Retrieving..." : "Get recommendation"}
        </button>
      </div>

      {error && <div className="error-box">{error}</div>}

      {result && (
        <div className="rehab-result">
          <div className="card">
            <div className="card-title">
              {result.used_llm ? "🧠 LLM-synthesized" : "📄 Retrieval-only (Ollama unavailable)"} — KL {result.kl_grade} ({result.label})
            </div>
            <p>{result.synthesis}</p>
          </div>

          <details>
            <summary>Retrieved {result.excerpts.length} guideline excerpt(s)</summary>
            {result.excerpts.map((chunk, i) => (
              <div className="excerpt" key={i}>
                <strong>{chunk.heading}</strong> <span className="excerpt-source">{chunk.source}</span>
                <p>{chunk.text}</p>
              </div>
            ))}
          </details>

          <p className="disclaimer">{result.disclaimer}</p>
        </div>
      )}
    </div>
  );
}
