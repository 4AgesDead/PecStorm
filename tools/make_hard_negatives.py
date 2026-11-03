"""Hard-negative кадры: где грузовика нет, а детектор срабатывает.

Идея: если GT-пивот **не попадает в кадр**, то любая детекция на этом кадре —
ложное срабатывание. Такие кадры добавляются в обучение как фон (пустой
label-файл), чтобы модель перестала реагировать на похожие объекты.

Не путать с пропусками: если пивот в кадре, а детекции нет — это miss, такие
кадры не берём.

Запуск:
    python tools/make_hard_negatives.py --detections out/truck/detections_ft.csv `
        --out datasets/truck/pseudo_hard --max-images 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np

from multiview3d.camera import project_point
from multiview3d.truck_dataset import DEFAULT_REPAIRS, DEFAULT_ROOT, TruckDataset
from multiview3d.truck_pipeline import load_detections


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Hard-negative кадры для дообучения")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--repairs", default=str(DEFAULT_REPAIRS))
    ap.add_argument("--detections", required=True)
    ap.add_argument("--out", default="datasets/truck/pseudo_hard")
    ap.add_argument("--max-images", type=int, default=500)
    ap.add_argument("--min-score", type=float, default=0.30)
    ap.add_argument("--jpeg-quality", type=int, default=92)
    args = ap.parse_args(argv)

    ds = TruckDataset.discover(args.root, repairs_path=args.repairs or None)
    gt = ds.ground_truth_by_frame()
    dets = load_detections(args.detections)
    out = Path(args.out)
    img_dir, lbl_dir = out / "images", out / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    picks: dict[str, list[tuple[int, float]]] = {c.name: [] for c in ds.cameras}
    for f, per_cam in dets.items():
        if f not in gt:
            continue
        for cam_name, items in per_cam.items():
            if cam_name not in picks:
                continue
            cam = ds.provider.calibrations_at(f, nearest=True).get(cam_name)
            if cam is None:
                continue
            try:
                uv = project_point(cam, gt[f])
            except ValueError:
                continue
            if 0 <= uv[0] < cam.width and 0 <= uv[1] < cam.height:
                continue                      # пивот в кадре: это не негатив
            best = max((d for d in items if d.bbox is not None),
                       key=lambda d: d.score, default=None)
            if best is None or best.score < args.min_score:
                continue
            picks[cam_name].append((f, float(best.score)))

    written = 0
    list_paths: list[str] = []
    for cam in ds.cameras:
        rows = sorted(picks[cam.name], key=lambda t: -t[1])
        if not rows:
            print(f"  {cam.name}: негативов нет")
            continue
        cap = cv2.VideoCapture(str(cam.video))
        if not cap.isOpened():
            print(f"  {cam.name}: не открыть видео", file=sys.stderr)
            continue
        n = 0
        for f, score in rows:
            if written >= args.max_images:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if not ok:
                continue
            stem = f"neg_{cam.name}_f{f:04d}"
            cv2.imwrite(str(img_dir / f"{stem}.jpg"), img,
                        [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
            (lbl_dir / f"{stem}.txt").write_text("", encoding="utf-8")
            list_paths.append((img_dir / f"{stem}.jpg").resolve().as_posix())
            written += 1
            n += 1
        cap.release()
        print(f"  {cam.name}: {n} негативов (score>= {args.min_score})")

    (out / "negatives.txt").write_text("\n".join(list_paths) + "\n", encoding="utf-8")

    # Собираем общий список train (основной датасет + негативы) и data-конфиг
    base = Path("datasets/truck/pseudo")
    train_base = (base / "train.txt").read_text(encoding="utf-8").splitlines()
    combined = [p for p in train_base if p.strip()] + list_paths
    (base / "train_with_neg.txt").write_text("\n".join(combined) + "\n", encoding="utf-8")
    yaml_text = (
        "# Сгенерировано tools/make_hard_negatives.py\n"
        f"path: {base.resolve().as_posix()}\n"
        "train: train_with_neg.txt\n"
        "val: val.txt\n"
        "names:\n  0: truck\n"
    )
    (base / "data_hard.yaml").write_text(yaml_text, encoding="utf-8")
    print(f"негативов всего: {written}; список: {out / 'negatives.txt'}")
    print(f"конфиг обучения: {base / 'data_hard.yaml'} "
          f"(train {len(combined)} изображений)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
