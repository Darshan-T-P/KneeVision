import { useEffect, useState } from "react";
import { getModelsStatus } from "./api";
import XrayPage from "./pages/XrayPage";
import ClinicalPage from "./pages/ClinicalPage";
import FusionPage from "./pages/FusionPage";
import RehabPage from "./pages/RehabPage";
import "./App.css";

const TABS = [
  { key: "xray", label: "🩻 X-ray Diagnosis", Component: XrayPage },
  { key: "clinical", label: "📝 Clinical Text", Component: ClinicalPage },
  { key: "fusion", label: "🔀 Fusion", Component: FusionPage },
  { key: "rehab", label: "🏃 Rehab", Component: RehabPage },
];

export default function App() {
  const [tab, setTab] = useState("xray");
  const [status, setStatus] = useState(null);
  const [apiDown, setApiDown] = useState(false);

  useEffect(() => {
    getModelsStatus().then(setStatus).catch(() => setApiDown(true));
  }, []);

  const ActiveComponent = TABS.find((t) => t.key === tab).Component;

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">🦵 KneeVision++</div>
        <div className="brand-sub">Multimodal knee OA diagnosis</div>
        <nav>
          {TABS.map((t) => (
            <button key={t.key} className={t.key === tab ? "nav-item active" : "nav-item"} onClick={() => setTab(t.key)}>
              {t.label}
            </button>
          ))}
        </nav>
        <div className="sidebar-footer">
          {apiDown && <div className="status-badge status-down">API unreachable</div>}
          {status && (
            <div className="status-badge status-up">
              API online · {status.image_models.length} image model(s)
              {status.clinical_available ? " · clinical ✓" : ""}
              {status.fusion_available ? " · fusion ✓" : ""}
            </div>
          )}
        </div>
      </aside>
      <main className="content">
        <ActiveComponent />
      </main>
    </div>
  );
}
