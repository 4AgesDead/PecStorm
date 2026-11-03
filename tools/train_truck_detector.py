"""Дообучение детектора на псевдо-разметке этой сцены (ultralytics).

Данные готовит ``tools/make_pseudo_labels.py`` (боксы из GT + 3D-модели).
Скрипт только оборачивает обучение: фиксированные гиперпараметры, лог времени,
проверка валидации после прогона.

Запуск (CPU):
    python tools/train_truck_detector.py --model yolov8n.pt --epochs 10 --imgsz 640
Запуск на GPU (если доступен torch с CUDA/HIP):
    python tools/train_truck_detector.py --model yolov8n.pt --device 0
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Дообучение YOLO на псевдо-разметке")
    ap.add_argument("--data", default="datasets/truck/pseudo/data.yaml")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cpu", help="cpu | 0 (CUDA/HIP)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--lr0", type=float, default=0.01,
                    help="стартовый learning rate (для дообучения с best.pt — 0.001..0.003)")
    ap.add_argument("--name", default="truck_scene")
    ap.add_argument("--project", default="out/train")
    ap.add_argument("--time-limit", type=float, default=None,
                    help="секунд на обучение (0 — без ограничения)")
    args = ap.parse_args(argv)

    if not Path(args.data).exists():
        print(f"нет {args.data}: сначала сделайте псевдо-разметку "
              f"(tools/make_pseudo_labels.py --extract-images)", file=sys.stderr)
        return 2

    from ultralytics import YOLO

    model = YOLO(args.model)
    t0 = time.perf_counter()
    results = model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, device=args.device, workers=args.workers,
        patience=args.patience, project=args.project, name=args.name,
        lr0=args.lr0, exist_ok=True, verbose=True,
        time=(args.time_limit / 3600.0) if args.time_limit else None,
        amp=args.device != "cpu",
        plots=False,
    )
    dt = time.perf_counter() - t0
    best = Path(args.project) / args.name / "weights" / "best.pt"
    doc = {
        "model": args.model,
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "seconds": round(dt, 1),
        "best_weights": str(best) if best.exists() else None,
        "metrics": {
            "mAP50": float(results.box.map50) if getattr(results, "box", None) else None,
            "mAP50-95": float(results.box.map) if getattr(results, "box", None) else None,
        },
    }
    out = Path(args.project) / args.name / "train_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"обучение: {dt / 60:.1f} мин; веса: {best}")
    print(f"mAP50={doc['metrics']['mAP50']}, mAP50-95={doc['metrics']['mAP50-95']}")
    print(f"отчёт: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
