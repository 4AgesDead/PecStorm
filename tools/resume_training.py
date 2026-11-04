"""Продолжение обучения ultralytics с чекпойнта (resume).

Запуск:
    python tools/resume_training.py --weights runs/.../weights/last.pt --epochs 90
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Resume обучения YOLO")
    ap.add_argument("--weights", required=True, help="путь к last.pt прогона")
    ap.add_argument("--epochs", type=int, default=None,
                    help="новый предел эпох (по умолчанию — из чекпойнта)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    from ultralytics import YOLO

    w = Path(args.weights)
    if not w.exists():
        print(f"нет файла {w}", file=sys.stderr)
        return 2
    model = YOLO(str(w))
    kwargs = {"resume": True}
    if args.epochs:
        kwargs["epochs"] = args.epochs
    if args.device is not None:
        kwargs["device"] = args.device

    t0 = time.perf_counter()
    model.train(**kwargs)
    print(f"продолжение завершено за {(time.perf_counter() - t0) / 60:.1f} мин")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
