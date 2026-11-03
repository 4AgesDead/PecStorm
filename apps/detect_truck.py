"""Детекция грузовика на пяти видео набора «Пеленга» -> CSV 2D-наблюдений.

Запуск (пример полного прогона):
    python apps/detect_truck.py --start 0 --end 1000 --step 1
        --model yolov8x.pt --conf 0.25 --imgsz 960
        --out out/truck/detections.csv --stats out/truck/detect_stats.json

Детектор — ML (ultralytics YOLO), класс ``truck`` (COCO). Фоновое вычитание
как основной детектор не используется: на статичных камерах оно даёт маску
дороги/облаков, а не объект (решение зафиксировано в docs/TRUCK_DATASET.md).

Выход: CSV ``camera,frame,timestamp,score,label,x1,y1,x2,y2,cx,cy``
и JSON статистики (покрытие по камерам, время на кадр).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from multiview3d.detect import build_detector
from multiview3d.truck_dataset import DEFAULT_REPAIRS, DEFAULT_ROOT, TruckDataset
from multiview3d.truck_pipeline import (
    detect_frames,
    detection_rows,
    frames_range,
    save_detections,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Детекция грузовика (YOLO) по датасету «Пеленга»")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--repairs", default=str(DEFAULT_REPAIRS))
    ap.add_argument("--start", type=int, default=None, help="первый кадр (по умолчанию общая область)")
    ap.add_argument("--end", type=int, default=None, help="последний кадр включительно")
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--model", default="yolov8x.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--class-names", default="truck,car,bus,train",
                    help="имена классов COCO через запятую (пусто — все классы)")
    ap.add_argument("--out", default="out/truck/detections.csv")
    ap.add_argument("--stats", default="out/truck/detect_stats.json")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args(argv)

    ds = TruckDataset.discover(args.root, repairs_path=args.repairs or None)
    frames = frames_range(ds, start=args.start, end=args.end, step=args.step)
    if args.max_frames:
        frames = frames[:args.max_frames]
    names = [n.strip() for n in args.class_names.split(",") if n.strip()]
    spec = {"kind": "yolo", "model": args.model, "conf": args.conf,
            "imgsz": args.imgsz, "device": args.device,
            "class_names": names or None}
    detector = build_detector(spec)

    print(f"Детектор: {args.model} (conf={args.conf}, imgsz={args.imgsz}, классы={names or 'все'})")
    print(f"Кадры: {frames[0]}..{frames[-1]} шаг {args.step} -> {len(frames)} кадров × "
          f"{len(ds.cameras)} камер")
    t0 = time.perf_counter()
    detections, stats = detect_frames(ds, detector, frames, log_every=args.log_every)
    wall = time.perf_counter() - t0

    per_cam_frames = {c.name: 0 for c in ds.cameras}
    for per_cam in detections.values():
        for name, dets in per_cam.items():
            if dets:
                per_cam_frames[name] = per_cam_frames.get(name, 0) + 1
    n = save_detections(detection_rows(detections), args.out)

    report = {
        "dataset": "pelenga_truck",
        "detector": spec,
        "frames_requested": len(frames),
        "frame_range": [frames[0], frames[-1]],
        "step": args.step,
        "frames_processed": stats.frames,
        "frames_with_detection": stats.frames_with_detection,
        "frame_detection_rate": (stats.frames_with_detection / max(stats.frames, 1)),
        "detections_total": stats.detections,
        "detections_per_frame": stats.detections / max(stats.frames, 1),
        "per_camera_detections": stats.per_camera,
        "per_camera_frames_with_detection": per_cam_frames,
        "per_camera_frame_rate": {k: v / max(stats.frames, 1)
                                  for k, v in per_cam_frames.items()},
        "seconds": round(wall, 2),
        "ms_per_frame": round(1000.0 * wall / max(stats.frames, 1), 1),
        "fps_processing": round(stats.frames / max(wall, 1e-9), 3),
        "detections_csv": str(args.out).replace("\\", "/"),
    }
    p = Path(args.stats)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Кадров обработано: {stats.frames}, с детекцией: {stats.frames_with_detection} "
          f"({100.0 * report['frame_detection_rate']:.1f} %)")
    print(f"Детекций всего: {stats.detections} ({report['detections_per_frame']:.2f}/кадр)")
    for name in ds.names:
        print(f"  {name}: {per_cam_frames[name]} кадров с детекцией "
              f"({100.0 * per_cam_frames[name] / max(stats.frames, 1):.1f} %), "
              f"детекций {stats.per_camera.get(name, 0)}")
    print(f"Время: {wall:.1f} с ({report['ms_per_frame']} мс/кадр, "
          f"{report['fps_processing']:.2f} кадр/с)")
    print(f"CSV: {args.out} ({n} строк); статистика: {args.stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
