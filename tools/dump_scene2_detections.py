"""Все детекции класса truck в кадрах scene2 (для ручного разбора начала)."""

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from ultralytics import YOLO

from multiview3d.scene2_dataset import Scene2Dataset

frames = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "0,50,100,140").split(",")]
ds = Scene2Dataset.discover("scene2")
model = YOLO("yolov8x.pt")
caps = {c.name: cv2.VideoCapture(str(c.video)) for c in ds.cameras}
for f in frames:
    print(f"--- кадр {f}")
    for name in ds.names:
        cap = caps[name]
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        res = model.predict(img, verbose=False, conf=0.2, imgsz=960, classes=[7])[0]
        boxes = [[int(v) for v in b.xyxy[0]] + [round(float(b.conf[0]), 2)] for b in res.boxes]
        if boxes:
            print("   ", name, boxes)
for cap in caps.values():
    cap.release()
