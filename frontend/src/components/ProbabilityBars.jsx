export default function ProbabilityBars({ probabilities }) {
  if (!probabilities) return null;
  const entries = Object.entries(probabilities);
  const max = Math.max(...Object.values(probabilities), 0.01);
  return (
    <div className="prob-bars">
      {entries.map(([label, value]) => (
        <div className="prob-row" key={label}>
          <span className="prob-label">{label}</span>
          <div className="prob-track">
            <div className="prob-fill" style={{ width: `${(value / max) * 100}%` }} />
          </div>
          <span className="prob-value">{(value * 100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  );
}
