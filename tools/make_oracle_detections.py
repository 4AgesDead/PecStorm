"""Оракульные детекции из псевдо-разметки: верхняя граница конвейера.

Берём боксы, построенные из GT-позиции и 3D-модели (файлы
``datasets/truck/pseudo/labels``), и подаём их как «идеальный детектор»
(score=1.0). Так видно, какое покрытие и MAE достижимы, если детектор будет
находить объект всегда и точно. Это не решение задачи, а измерение потолка.
"""

import csv
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

W, H = 1920, 1080
root = Path("datasets/truck/pseudo")
rows = []
for lbl in sorted((root / "labels").glob("*.txt")):
    text = lbl.read_text(encoding="utf-8").strip()
    if not text:
        continue
    parts = text.split()
    if len(parts) != 5:
        continue
    _, cx, cy, bw, bh = parts
    cam, frame = lbl.stem.split("_f")
    f = int(frame)
    cxp, cyp = float(cx) * W, float(cy) * H
    wp, hp = float(bw) * W, float(bh) * H
    x1, y1 = cxp - wp / 2, cyp - hp / 2
    x2, y2 = cxp + wp / 2, cyp + hp / 2
    rows.append({"camera": cam, "frame": f, "timestamp": f / 30.0, "score": 1.0,
                 "label": "0", "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                 "cx": cxp, "cy": cyp})

rows.sort(key=lambda r: (r["frame"], r["camera"]))
out = Path("out/truck/detections_oracle.csv")
with out.open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["camera", "frame", "timestamp", "score", "label",
                                       "x1", "y1", "x2", "y2", "cx", "cy"])
    w.writeheader()
    w.writerows(rows)
print(f"оракульных боксов: {len(rows)} -> {out}")
