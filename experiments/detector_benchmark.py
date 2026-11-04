"""Бенчмарк детектора: модели/разрешения/пороги на кадрах набора «Пеленга».

Запуск:
    python experiments/detector_benchmark.py [--step 100]
        [--models yolov8n.pt,yolov8x.pt] [--imgsz 640,960] [--conf 0.25]

Метрики: доля кадров с детекцией по камерам (detection rate), число детекций
на кадр, средняя/медианная уверенность, время на кадр. Ground truth 2D нет,
поэтому precision/recall здесь не считаются: только оценка recall прокси по
доле кадров с детекцией (обсуждается в docs/DETECTION.md).

Отчёт: ``out/experiments/detector_benchmark.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from multiview3d.detect import YoloDetector
from multiview3d.truck_dataset import TruckDataset, TruckVideoSource

CLASS_NAMES = ["truck", "car", "bus", "train"]


def benchmark(ds: TruckDataset, model: str, imgsz: int, conf: float,
              frames: list[int]) -> dict:
    det = YoloDetector(model, conf=conf, imgsz=imgsz, class_names=CLASS_NAMES)
    per_cam = {n: {"frames": 0, "with_det": 0, "scores": []} for n in ds.names}
    t0 = time.perf_counter()
    sources = {c.name: TruckVideoSource(c, start=frames[0], end=frames[-1], fps=ds.fps)
               for c in ds.cameras}
    try:
        for frame in frames:
            for name, src in sources.items():
                while src.frame_index < frame:
                    if src.read() is None:
                        raise RuntimeError(f"{name}: не дочитан до кадра {frame}")
                item = src.read()
                if item is None:
                    continue
                _, image, _ = item
                dets = det.detect(image)
                st = per_cam[name]
                st["frames"] += 1
                if dets:
                    st["with_det"] += 1
                    st["scores"].append(max(d.score for d in dets))
    finally:
        for src in sources.values():
            src.close()
    wall = time.perf_counter() - t0
    total_frames = sum(v["frames"] for v in per_cam.values())
    return {
        "model": model,
        "imgsz": imgsz,
        "conf": conf,
        "frames_total": total_frames,
        "seconds": round(wall, 2),
        "ms_per_camera_frame": round(1000.0 * wall / max(total_frames, 1), 1),
        "detection_rate_overall": round(
            sum(v["with_det"] for v in per_cam.values()) / max(total_frames, 1), 4),
        "per_camera": {
            n: {
                "frames": v["frames"],
                "frames_with_detection": v["with_det"],
                "detection_rate": round(v["with_det"] / max(v["frames"], 1), 4),
                "score_mean": (round(float(np.mean(v["scores"])), 4) if v["scores"] else None),
                "score_median": (round(float(np.median(v["scores"])), 4)
                                 if v["scores"] else None),
            }
            for n, v in per_cam.items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Бенчмарк YOLO на датасете «Пеленга»")
    ap.add_argument("--root", default="scene1/zadania")
    ap.add_argument("--step", type=int, default=100)
    ap.add_argument("--models", default="yolov8n.pt,yolov8x.pt")
    ap.add_argument("--imgsz", default="640,960")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--max-frames", type=int, default=11)
    ap.add_argument("--out", default="out/experiments")
    args = ap.parse_args(argv)

    ds = TruckDataset.discover(args.root)
    frames = list(range(0, ds.frame_count(), args.step))[:args.max_frames]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    sizes = [int(s) for s in args.imgsz.split(",") if s.strip()]

    runs = []
    for model in models:
        for imgsz in sizes:
            print(f"--- {model} imgsz={imgsz} conf={args.conf}: {len(frames)} кадров × "
                  f"{len(ds.cameras)} камер")
            r = benchmark(ds, model, imgsz, args.conf, frames)
            runs.append(r)
            print(f"    detection rate {100.0 * r['detection_rate_overall']:.1f} %, "
                  f"{r['ms_per_camera_frame']} мс/кадр-камера")
            for name, s in r["per_camera"].items():
                print(f"      {name}: {100.0 * s['detection_rate']:.0f} %"
                      + (f", conf~{s['score_median']}" if s["score_median"] else ""))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    doc = {"frames": [frames[0], frames[-1], args.step],
           "note": ("detection rate — прокси recall: 2D-разметки нет, precision "
                    "не измеряется; классы COCO " + ",".join(CLASS_NAMES)),
           "runs": runs}
    (out / "detector_benchmark.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"JSON: {out / 'detector_benchmark.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
