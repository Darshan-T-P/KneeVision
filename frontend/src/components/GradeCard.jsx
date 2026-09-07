const KL_COLORS = ["#22c55e", "#84cc16", "#eab308", "#f97316", "#dc2626"];

export default function GradeCard({ title, prediction }) {
  if (!prediction) return null;
  const { kl_grade: klGrade, label, confidence } = prediction;
  return (
    <div className="card grade-card">
      <div className="card-title">{title}</div>
      <div className="grade-value" style={{ color: KL_COLORS[klGrade] }}>
        KL {klGrade}
      </div>
      <div className="grade-label">{label}</div>
      <div className="bar-track">
        <div className="bar-fill" style={{ width: `${confidence * 100}%`, background: KL_COLORS[klGrade] }} />
      </div>
      <div className="grade-sub">confidence {(confidence * 100).toFixed(1)}%</div>
    </div>
  );
}
