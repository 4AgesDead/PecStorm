"""Псевдо-разметка scene2 из COCO yolov8x для дообучения детектора.

Сохраняет полноразмерные кадры и YOLO-метки (класс 0 = truck) в
`datasets/scene2_pseudo/{images,labels}/{train,val}`. Разделение по времени:
train < 1500, val >= 1500.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from ultralytics import YOLO

from multiview3d.scene2_dataset import Scene2Dataset


def frames_to_run() -> list[int]:
    fs = set(range(0, 1932, 10))
    fs.update(range(350, 651, 5))       # больше цели
    fs.update(range(1040, 1300, 5))
    return sorted(fs)


def main() -> int:
    ds = Scene2Dataset.discover("scene2")
    model = YOLO("yolov8x.pt")
    root = Path("datasets/scene2_pseudo")
    for part in ("train", "val"):
        (root / "images" / part).mkdir(parents=True, exist_ok=True)
        (root / "labels" / part).mkdir(parents=True, exist_ok=True)
    frames = frames_to_run()
    counts = {"train": 0, "val": 0}
    boxes = 0
    for k, f in enumerate(frames):
        part = "train" if f < 1500 else "val"
        for cam in ds.cameras:
            cap = cv2.VideoCapture(str(cam.video))
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            cap.release()
            if not ok:
                continue
            r = model.predict(img, verbose=False, conf=0.2, imgsz=960,
                              device=0, classes=[7])[0]
            if len(r.boxes) == 0:
                continue
            h, w = img.shape[:2]
            lines = []
            for b in r.boxes:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                bw, bh = (x2 - x1) / w, (y2 - y1) / h
                lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            name = f"{cam.name}_{f:04d}"
            cv2.imwrite(str(root / "images" / part / (name + ".jpg")), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
            (root / "labels" / part / (name + ".txt")).write_text(
                "\n".join(lines) + "\n", encoding="utf-8")
            counts[part] += 1
            boxes += len(lines)
        if k % 20 == 0:
            print(f"{k}/{len(frames)} кадров, изображений {counts}, боксов {boxes}")
    (root / "scene2.yaml").write_text(
        f"path: {root.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\nnames:\n  0: truck\n",
        encoding="utf-8")
    print(f"готово: {root}; images={counts}, boxes={boxes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
