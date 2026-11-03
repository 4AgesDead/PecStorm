"""Калибрация камер по изображениям полигона (шахматного или кругового).

Запуск (один каталог = одна камера, все камеры решаются в системе координат
полигона, поэтому согласованы между собой):
    python -m apps.calibrate --images data/cam1 data/cam2 data/cam3 \
        --pattern 9x6 --square 30 --out config/calib.json

Имена камер берутся из имён каталогов (или задаются ``--names``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import json

from multiview3d.calibration import (
    calibrate_camera_from_images,
    load_calibration,
    quality_report,
    save_calibration,
)


def parse_pattern(s: str) -> tuple[int, int]:
    parts = s.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"pattern должен быть вида '9x6', получено {s!r}")
    return (int(parts[0]), int(parts[1]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Калибрация камер по фото полигона")
    ap.add_argument("--images", nargs="+", required=True,
                    help="Каталоги снимков: по одному на камеру")
    ap.add_argument("--names", nargs="+", default=None, help="Имена камер (по числу каталогов)")
    ap.add_argument("--pattern", default="9x6", help="Внутренние углы полигона, напр. 9x6")
    ap.add_argument("--square", type=float, default=30.0, help="Сторона клетки, мм")
    ap.add_argument("--kind", choices=["chessboard", "circles"], default="chessboard")
    ap.add_argument("--out", default="config/calib.json")
    ap.add_argument("--update", action="store_true",
                    help="Добавить камеры к существующему файлу вместо перезаписи")
    args = ap.parse_args(argv)

    pattern = parse_pattern(args.pattern)
    dirs = [Path(p) for p in args.images]
    names = args.names or [d.name for d in dirs]
    if len(names) != len(dirs):
        print("Ошибка: --names должен совпадать по длине с --images", file=sys.stderr)
        return 2

    cams, meta = ([], {})
    if args.update and Path(args.out).exists():
        cams, meta = load_calibration(args.out)

    reports: dict[str, float] = {}
    for i, (d, name) in enumerate(zip(dirs, names)):
        images = sorted(p for p in d.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"))
        if not images:
            print(f"Ошибка: в {d} нет изображений", file=sys.stderr)
            return 2
        try:
            cam, info = calibrate_camera_from_images(name, images, pattern,
                                                     args.square, args.kind)
        except Exception as exc:
            print(f"Ошибка камеры {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        cams.append(cam)
        reports[len(cams) - 1] = info["reprojection_rms_px"]
        print(f"{name}: RMS репроекции {info['reprojection_rms_px']:.2f} px, "
              f"ракурсов {info['views_used']}/{len(images)}")

    save_calibration(args.out, cams, meta={**meta, "pattern": args.pattern,
                                           "square_mm": args.square, "kind": args.kind})
    rep = quality_report(cams, reproj_rms_px=reports)
    print("Базисы камер (в единицах полигона):", rep["baselines"])
    for w in rep["warnings"]:
        print("Предупреждение:", w)
    print(f"Сохранено {len(cams)} камер в {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
