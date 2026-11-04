"""Собрать детекции COCO для fit модели scene2 (только класс truck)."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
from ultralytics import YOLO

from multiview3d.scene2_dataset import Scene2Dataset

OUT = Path("out/scene2/dets_fit.csv")


def frames_to_run() -> list[int]:
    fs = set(range(0, 1932, 50))
    for a, b, step in ((160, 270, 10), (360, 460, 10), (690, 735, 5),
                       (1040, 1145, 10), (1160, 1300, 10),
                       (1540, 1710, 10), (1750, 1850, 10)):
        fs.update(range(a, b + 1, step))
    return sorted(fs)


def main() -> int:
    ds = Scene2Dataset.discover("scene2")
    model = YOLO("yolov8x.pt")
    frames = frames_to_run()
    rows = []
    for k, f in enumerate(frames):
        for cam in ds.cameras:
            cap = cv2.VideoCapture(str(cam.video))
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            cap.release()
            if not ok:
                continue
            r = model.predict(img, verbose=False, conf=0.2, imgsz=960,
                              device=0, classes=[7])[0]
            for b in r.boxes:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                rows.append({"frame": f, "camera": cam.name,
                             "x1": round(x1, 2), "y1": round(y1, 2),
                             "x2": round(x2, 2), "y2": round(y2, 2),
                             "score": round(float(b.conf[0]), 4)})
        if k % 10 == 0:
            print(f"{k}/{len(frames)} кадров, детекций {len(rows)}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"готово: {OUT}, {len(rows)} детекций")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
