"""Датасет кропов для корректора центра бокса (эксперимент, не основной код).

Для каждого кадра/камеры берётся лучшая детекция, вокруг неё — квадратный кроп
(``crop_scale`` × max(w, h)), цель — проекция GT-пивота. В CSV пишется
нормализованное смещение цели от центра кропа.

Сплиты по времени, чтобы не было утечки:
* train: 0..539 и 581..699 (включая пропуски 45..59 и 386..427);
* val_gap: 540..580 (включая НЕвиденный пропуск 551..572);
* holdout: 700..1000 (не трогаем до финальной проверки).

Запуск:
    py experiments/build_center_dataset.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np

from multiview3d.camera import project_point
from multiview3d.truck_dataset import DEFAULT_REPAIRS, DEFAULT_ROOT, TruckDataset

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def load_best(path: str | Path) -> dict[tuple[int, str], dict]:
    best: dict[tuple[int, str], dict] = {}
    with Path(path).open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            key = (int(r["frame"]), r["camera"])
            cur = best.get(key)
            if cur is None or float(r["score"]) > float(cur["score"]):
                best[key] = r
    return best


def split_of(frame: int) -> str | None:
    if 0 <= frame <= 539 or 581 <= frame <= 699:
        return "train"
    if 540 <= frame <= 580:
        return "val_gap"
    if 700 <= frame <= 1000:
        return "holdout"
    return None


def crop_around(frame: np.ndarray, cx: float, cy: float, side: float,
                out_size: int) -> np.ndarray:
    h, w = frame.shape[:2]
    half = side / 2.0
    x0, y0 = int(round(cx - half)), int(round(cy - half))
    x1, y1 = x0 + int(round(side)), y0 + int(round(side))
    left, top = max(0, -x0), max(0, -y0)
    right, bottom = max(0, x1 - w), max(0, y1 - h)
    crop = frame[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    if left or top or right or bottom:
        crop = cv2.copyMakeBorder(crop, top, bottom, left, right,
                                  cv2.BORDER_REPLICATE)
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Кропы «бокс → GT-пивот»")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--repairs", default=str(DEFAULT_REPAIRS))
    ap.add_argument("--detections", default="out/truck/det_ens_v4b_v2v1.csv")
    ap.add_argument("--out", default="out/experiment_kp")
    ap.add_argument("--crop-scale", type=float, default=2.0)
    ap.add_argument("--size", type=int, default=192)
    ap.add_argument("--min-score", type=float, default=0.5,
                    help="не учиться на низкоуверенных боксах (обычно ложные)")
    ap.add_argument("--jpeg-quality", type=int, default=90)
    args = ap.parse_args(argv)

    ds = TruckDataset.discover(args.root, repairs_path=args.repairs or None)
    gt = ds.ground_truth_by_frame()
    dets = load_best(args.detections)
    out = Path(args.out)
    for sub in ("train", "val_gap", "holdout"):
        (out / "crops" / sub).mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    skipped = {"no_gt": 0, "no_det": 0, "project": 0, "outside": 0, "low_score": 0}
    for cam in ds.cameras:
        samples = [(f, dets[(f, cam.name)]) for f in range(0, 1001)
                   if (f, cam.name) in dets and f in gt]
        samples.sort(key=lambda x: x[0])
        cap = cv2.VideoCapture(str(cam.video))
        if not cap.isOpened():
            print(f"{cam.name}: не открыть видео", file=sys.stderr)
            continue
        by_frame = dict(samples)
        last = max((f for f in by_frame), default=-1)
        f = 0
        while f <= last:
            ok, frame = cap.read()
            if not ok:
                break
            det = by_frame.get(f)
            if det is not None:
                split = split_of(f)
                if split is None:
                    f += 1
                    continue
                if float(det.get("score", 1.0)) < args.min_score:
                    skipped["low_score"] += 1
                    f += 1
                    continue
                calib = ds.provider.calibrations_at(f, nearest=True).get(cam.name)
                if calib is None:
                    skipped["project"] += 1
                    f += 1
                    continue
                x1, y1, x2, y2 = (float(det[k]) for k in ("x1", "y1", "x2", "y2"))
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                side = max(args.crop_scale * max(x2 - x1, y2 - y1), 64.0)
                try:
                    u, v = project_point(calib, gt[f])
                except (ValueError, np.linalg.LinAlgError):
                    skipped["project"] += 1
                    f += 1
                    continue
                if not (np.isfinite(u) and np.isfinite(v)):
                    skipped["project"] += 1
                    f += 1
                    continue
                if abs(u - cx) > side / 2 - 4 or abs(v - cy) > side / 2 - 4:
                    skipped["outside"] += 1
                    f += 1
                    continue
                crop = crop_around(frame, cx, cy, side, args.size)
                name = f"{split}_{f:04d}_{cam.name}.jpg"
                path = out / "crops" / split / name
                cv2.imwrite(str(path), crop,
                            [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                rows.append({
                    "split": split, "frame": f, "camera": cam.name,
                    "path": str(path).replace("\\", "/"),
                    "crop_cx": f"{cx:.3f}", "crop_cy": f"{cy:.3f}",
                    "side": f"{side:.3f}",
                    "label_x": f"{(u - cx) / side:.6f}",
                    "label_y": f"{(v - cy) / side:.6f}",
                    "baseline_px": f"{np.hypot(u - cx, v - cy):.3f}",
                    "pivot_u": f"{u:.3f}", "pivot_v": f"{v:.3f}",
                    "box_w": f"{x2 - x1:.1f}", "box_h": f"{y2 - y1:.1f}",
                })
            f += 1
        cap.release()

    csv_path = out / "samples.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["split"]] = counts.get(r["split"], 0) + 1
    print(f"кропов: {len(rows)} -> {csv_path}")
    print(f"по сплитам: {counts}")
    print(f"пропущено: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
