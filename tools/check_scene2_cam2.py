"""Визуальная проверка: проекция итоговой точки на аэрокадр cam2."""

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from multiview3d.camera import project_point
from multiview3d.scene2_dataset import Scene2Dataset

SUB = sys.argv[1] if len(sys.argv) > 1 else "out/scene2/submission_try3.csv"
FRAMES = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else
                           ["470", "520", "560", "620"])]

ds = Scene2Dataset.discover("scene2")
pos = {}
for r in csv.DictReader(open(SUB, encoding="utf-8")):
    pos[int(r["frame"])] = (np.array([float(r["x"]), float(r["y"]), float(r["z"])]),
                            r["status"])

cap = cv2.VideoCapture(str([c for c in ds.cameras if c.name == "cam2"][0].video))
for f in FRAMES:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, img = cap.read()
    if not ok:
        continue
    cam = ds.provider.calibration("cam2", f)
    p, st = pos[f]
    u, v = project_point(cam, p)
    canvas = img.copy()
    if np.isfinite(u) and np.isfinite(v):
        cv2.drawMarker(canvas, (int(u), int(v)), (0, 0, 255), cv2.MARKER_CROSS, 60, 3)
        x0, y0 = int(max(0, u - 260)), int(max(0, v - 200))
        x1, y1 = int(min(1919, u + 260)), int(min(1079, v + 200))
        crop = canvas[y0:y1, x0:x1]
    else:
        crop = canvas
    out = Path(f"out/scene2/check_cam2_{f:04d}.png")
    cv2.imwrite(str(out), crop)
    print(out, st, "проекция", None if not np.isfinite(u) else (round(u), round(v)))
cap.release()
