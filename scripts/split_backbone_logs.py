"""Split logs/backbone_queue.log into one clean log per backbone (logs/backbones/<model>.log).

Drops tqdm progress-bar noise; each file holds that model's training epochs plus its
single test evaluation. Safe to re-run at any time (read-only on the source log).
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "logs" / "backbone_queue.log"
out = ROOT / "logs" / "backbones"
out.mkdir(exist_ok=True)

lines = src.read_text(errors="replace").replace("\r", "\n").split("\n")
segments: dict[str, list[str]] = {}
current = "queue_setup"
for line in lines:
    if re.search(r"it/s\]|s/it\]|^Training:|^Validation:", line):
        continue
    m = re.search(r"Training: (\S+)\s+\|\s+Ordinal", line)
    if m:
        current = m.group(1)
    if line.strip():
        segments.setdefault(current, []).append(line)
for name, body in segments.items():
    (out / f"{name}.log").write_text("\n".join(body) + "\n")
    print(f"{name}: {len(body)} lines -> logs/backbones/{name}.log")
