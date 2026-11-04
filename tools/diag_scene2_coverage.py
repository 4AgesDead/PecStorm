"""Диагностика покрытия scene2: сколько камер реально видят цель на кадре.

Вопрос, на который отвечает скрипт: текущие ~500 MEASURED — это потолок
детектора или потолок сопоставления/триангуляции?

Метод: берём текущую траекторию из submission CSV, проецируем точку в каждую
камеру и ищем детекцию COCO, рядом с которой эта проекция оказалась. Если
камер с детекцией >= 2, кадр в принципе измерим (при верной ассоциации).

Это оценка сверху: проекция опирается на интерполированную траекторию, а не
на GT. Поэтому результат — «сколько кадров потенциально измеримы», не более.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np

from multiview3d.camera import project_point
from multiview3d.scene2_dataset import Scene2Dataset


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Диагностика покрытия scene2")
    ap.add_argument("--submission", default="results/scene2/submission_scene2.csv")
    ap.add_argument("--model", default="yolov8x.pt")
    ap.add_argument("--every", type=int, default=20)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--near-px", type=float, default=250.0)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default="out/scene2/coverage_diag.csv")
    args = ap.parse_args(argv)

    from ultralytics import YOLO

    ds = Scene2Dataset.discover("scene2")
    model = YOLO(args.model)
    pos: dict[int, np.ndarray] = {}
    status: dict[int, str] = {}
    with Path(args.submission).open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            f = int(r["frame"])
            pos[f] = np.array([float(r["x"]), float(r["y"]), float(r["z"])])
            status[f] = r.get("status", "")
    frames = sorted(f for f in pos if f % args.every == 0)

    caps = {c.name: cv2.VideoCapture(str(c.video)) for c in ds.cameras}
    rows = []
    for k, f in enumerate(frames):
        imgs = {}
        for name, cap in caps.items():
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if ok:
                imgs[name] = img
        cams = ds.provider.calibrations_at(f, nearest=True)
        hits = {}
        for name, img in imgs.items():
            r = model.predict(img, verbose=False, conf=args.conf,
                              imgsz=args.imgsz, device=args.device, classes=[7])[0]
            if len(r.boxes) == 0:
                hits[name] = 0
                continue
            try:
                u, v = project_point(cams[name], pos[f])
            except ValueError:
                hits[name] = 0
                continue
            best = None
            for b in r.boxes:
                x1, y1, x2, y2 = [float(t) for t in b.xyxy[0]]
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                d = float(np.hypot(cx - u, cy - v))
                if d <= args.near_px and (best is None or d < best):
                    best = d
            hits[name] = 1 if best is not None else 0
        n_hit = sum(hits.values())
        rows.append({"frame": f, "status": status.get(f, ""),
                     "cameras": n_hit, **{f"hit_{n}": v for n, v in hits.items()}})
        if k % 10 == 0:
            print(f"{k}/{len(frames)} кадров", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    cnt = Counter(r["cameras"] for r in rows)
    print(f"кадров в выборке: {len(rows)}")
    print("распределение числа камер с детекцией:", dict(sorted(cnt.items())))
    ge2 = [r for r in rows if r["cameras"] >= 2]
    print(f"кадров с >=2 камерами: {len(ge2)} ({len(ge2) / len(rows) * 100:.1f}%)")
    for st in ("MEASURED", "INTERPOLATED"):
        sub = [r for r in rows if r["status"] == st]
        if sub:
            ok = sum(1 for r in sub if r["cameras"] >= 2)
            print(f"  {st}: {len(sub)} кадров, из них с >=2 камерами {ok} "
                  f"({ok / len(sub) * 100:.1f}%)")
    per_cam = {k: sum(r[f"hit_{k}"] for r in rows) for k in ds.names
               if f"hit_{k}" in rows[0]}
    print("детекций по камерам (кадров из выборки):", per_cam)
    print(f"сохранено: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
