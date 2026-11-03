"""Псевдо-разметка кадров по 3D-модели + GT: YOLO-датасет для дообучения.

Зачем: COCO-модель находит этот синтетический грузовик лишь на ~38 % видимых
камеро-кадров (см. docs/RESULTS.md §8a). Ручная разметка не нужна: GT даёт
позицию, 3D-модель (подогнанная к bbox, `tools/fit_truck_model.py`) — габариты
и ориентацию, калибровка — проекцию. Получаем боксы для всех кадров, где
модель целиком попадает в кадр.

Важно: это **учитель из GT**, а не независимая разметка. Точность боксов
ограничена подгонкой модели (IoU ≈0.54 в среднем, 0.74 на cam1) и курсом,
который считается по разности GT-позиций (ориентации в GT нет).

Запуск (сначала без картинок — только посчитать):
    python tools/make_pseudo_labels.py --limit 10
    python tools/make_pseudo_labels.py --limit 10 --extract-images
    python tools/make_pseudo_labels.py                    # полный набор
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np

from multiview3d.camera import project_point
from multiview3d.truck_dataset import DEFAULT_REPAIRS, DEFAULT_ROOT, TruckDataset
from tools.fit_truck_model import corners_world, fbx_size, heading_from_gt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Псевдо-разметка по 3D-модели и GT")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--repairs", default=str(DEFAULT_REPAIRS))
    ap.add_argument("--model", default="models/CargoTruck.fbx")
    ap.add_argument("--alignment", default="datasets/truck/model_alignment.json")
    ap.add_argument("--out", default="datasets/truck/pseudo")
    ap.add_argument("--min-box", type=float, default=24.0,
                    help="минимальная ширина бокса в пикселях")
    ap.add_argument("--margin", type=int, default=2,
                    help="запас до границы кадра (px); боксы у края пропускаются")
    ap.add_argument("--train-end", type=int, default=700)
    ap.add_argument("--extract-images", action="store_true")
    ap.add_argument("--jpeg-quality", type=int, default=92)
    ap.add_argument("--limit", type=int, default=None,
                    help="ограничить число кадров на камеру (для проверки)")
    args = ap.parse_args(argv)

    align = json.loads(Path(args.alignment).read_text(encoding="utf-8"))
    size = fbx_size(args.model) * float(align["scale"])
    offset = np.asarray(align["offset_m"], dtype=float)
    dyaw = np.deg2rad(float(align["dyaw_deg"]))
    up, fwd = align["axes"]["up"], align["axes"]["forward"]
    sign = float(align["axes"]["forward_sign"])
    cam_shift = {k: np.asarray(v, dtype=float)
                 for k, v in (align.get("camera_shift_px") or {}).items()}
    print(f"модель: масштаб {align['scale']:.3f} -> габариты "
          f"{np.round(size, 2)} м (в единицах сцены); "
          f"попиксельные сдвиги по камерам: {len(cam_shift)}")

    ds = TruckDataset.discover(args.root, repairs_path=args.repairs or None)
    gt = ds.ground_truth_by_frame()
    yaw_gt = heading_from_gt(gt)

    out = Path(args.out)
    img_dir, lbl_dir = out / "images", out / "labels"
    if args.extract_images:
        img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(gt)
    lo, hi = ds.common_frame_range()
    frames = [f for f in frames if lo <= f <= hi]   # GT идёт до 1001, видео — до 1000
    if args.limit:
        frames = frames[: args.limit]

    counts = {}
    train_list, val_list = [], []
    for cam in ds.cameras:
        name = cam.name
        cap = None
        n_lab = 0
        if args.extract_images:
            cap = cv2.VideoCapture(str(cam.video))
            if not cap.isOpened():
                print(f"ВНИМАНИЕ: не открыть видео {cam.video}", file=sys.stderr)
                cap = None
        for f in frames:
            calib = ds.provider.calibrations_at(f, nearest=True).get(name)
            if calib is None:
                continue
            center = np.asarray(gt[f], dtype=float) + offset
            pts = corners_world(size, center, yaw_gt[f] + dyaw, up, fwd, sign)
            uv = []
            ok = True
            for p in pts:
                try:
                    u, v = project_point(calib, p)
                except ValueError:
                    ok = False
                    break
                if not np.isfinite(u) or not np.isfinite(v):
                    ok = False
                    break
                uv.append((u, v))
            if not ok:
                continue
            us = [p[0] for p in uv]
            vs = [p[1] for p in uv]
            x1, x2 = min(us), max(us)
            y1, y2 = min(vs), max(vs)
            if name in cam_shift:
                x1 += cam_shift[name][0]
                x2 += cam_shift[name][0]
                y1 += cam_shift[name][1]
                y2 += cam_shift[name][1]
            m = args.margin
            if x1 < m or y1 < m or x2 > calib.width - m or y2 > calib.height - m:
                continue
            if (x2 - x1) < args.min_box:
                continue
            cx = (x1 + x2) / 2 / calib.width
            cy = (y1 + y2) / 2 / calib.height
            bw = (x2 - x1) / calib.width
            bh = (y2 - y1) / calib.height
            stem = f"{name}_f{f:04d}"
            (lbl_dir / f"{stem}.txt").write_text(
                f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n", encoding="utf-8")
            n_lab += 1
            (train_list if f <= args.train_end else val_list).append(stem)

            if cap is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                okf, img = cap.read()
                if okf:
                    cv2.imwrite(str(img_dir / f"{stem}.jpg"), img,
                                [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
        if cap is not None:
            cap.release()
        counts[name] = n_lab
        print(f"  {name}: размечено {n_lab} кадров")

    (out / "train.txt").write_text("\n".join(train_list) + "\n", encoding="utf-8")
    (out / "val.txt").write_text("\n".join(val_list) + "\n", encoding="utf-8")
    yaml_text = (
        f"# Сгенерировано tools/make_pseudo_labels.py\n"
        f"path: {out.resolve().as_posix()}\n"
        f"train: train.txt\nval: val.txt\n"
        f"names:\n  0: truck\n"
    )
    (out / "data.yaml").write_text(yaml_text, encoding="utf-8")
    total = sum(counts.values())
    print(f"итого размечено: {total} (train {len(train_list)}, val {len(val_list)})")
    print(f"датасет: {out} (data.yaml{', images/' if args.extract_images else ''})")
    if not args.extract_images:
        print("картинки не выгружались: добавьте --extract-images перед обучением")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
