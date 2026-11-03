"""Калибровка 2D-сдвига центра детекции по камерам (px) на train-кадрах.

Зачем: детектор даёт центр bbox, GT — pivot актора. Смещение «bbox center ↔
проекция pivot» систематическое и **разное по камерам** (ракурс, principal
point, ошибки intrinsics). Постоянный 3D-сдвиг (`anchor_offset_m`) компенсирует
только общее смещение; leave-one-camera-out проверка показывает, что на камеру
без данных фита он не переносится. Поэтому оценивается отдельный 2D-сдвиг
`(du, dv)` на камеру.

Метод: для каждого наблюдения (кадр с GT, лучшая детекция камеры) берётся
невязка `bbox_center - projection(pivot_gt)`. Сдвиг — устойчивая оценка центра
(медиана) по train-кадрам; он **вычитается** из центров детекций до
триангуляции (`FusionConfig.anchor_px_offset`).

Запуск:
    python tools/anchor_fit_px.py --detections out/truck/detections.csv `
        --train-start 0 --train-end 500 --holdout-start 600 --holdout-end 1000

Пишет ``datasets/truck/anchor_px_offset.json`` и печатает невязки
train/holdout до и после. GT используется только для калибровки на train,
на holdout — только для проверки.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from multiview3d.camera import project_point
from multiview3d.truck_dataset import TruckDataset
from multiview3d.truck_pipeline import load_detections


def residuals_by_camera(ds: TruckDataset, dets: dict, gt: dict[int, np.ndarray],
                        frames: list[int]) -> dict[str, np.ndarray]:
    """Невязки ``bbox_center - projection(pivot_gt)`` по камерам, (N,2)."""
    out: dict[str, list[np.ndarray]] = {n: [] for n in ds.names}
    for f in frames:
        if f not in gt:
            continue
        cams = ds.provider.calibrations_at(f, nearest=True)
        for name, cam_dets in (dets.get(f) or {}).items():
            cam_dets = [d for d in cam_dets if d.bbox is not None]
            if not cam_dets or name not in cams:
                continue
            best = max(cam_dets, key=lambda d: d.score)
            try:
                uv = project_point(cams[name], gt[f])
            except ValueError:
                continue
            out[name].append(np.asarray(best.center, dtype=float) - uv)
    return {n: np.asarray(v, dtype=float).reshape(-1, 2) for n, v in out.items()}


def stats(arr: np.ndarray) -> dict:
    if arr.size == 0:
        return {"n": 0}
    d = np.linalg.norm(arr, axis=1)
    return {"n": int(d.size), "mean_px": round(float(d.mean()), 2),
            "median_px": round(float(np.median(d)), 2),
            "p90_px": round(float(np.percentile(d, 90)), 2),
            "max_px": round(float(d.max()), 2)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="2D-сдвиг центра детекции по камерам")
    ap.add_argument("--root", default="huniashahatona/zadania")
    ap.add_argument("--detections", default="out/truck/detections.csv")
    ap.add_argument("--train-start", type=int, default=0)
    ap.add_argument("--train-end", type=int, default=500)
    ap.add_argument("--holdout-start", type=int, default=600)
    ap.add_argument("--holdout-end", type=int, default=1000)
    ap.add_argument("--min-observations", type=int, default=10)
    ap.add_argument("--out", default="datasets/truck/anchor_px_offset.json")
    args = ap.parse_args(argv)

    ds = TruckDataset.discover(args.root)
    gt = ds.ground_truth_by_frame()
    dets = load_detections(args.detections)
    train_frames = list(range(args.train_start, args.train_end + 1))
    hold_frames = list(range(args.holdout_start, args.holdout_end + 1))

    tr = residuals_by_camera(ds, dets, gt, train_frames)
    ho = residuals_by_camera(ds, dets, gt, hold_frames)

    offsets: dict[str, list[float]] = {}
    per_cam = {}
    for name in ds.names:
        a = tr[name]
        if a.shape[0] >= args.min_observations:
            offsets[name] = [round(float(v), 3) for v in np.median(a, axis=0)]
        corrected = ho[name] - np.asarray(offsets.get(name, [0.0, 0.0])) if ho[name].size else np.empty((0, 2))
        per_cam[name] = {
            "train": stats(a),
            "holdout_before": stats(ho[name]),
            "holdout_after": stats(corrected),
            "offset_px": offsets.get(name),
        }

    print(f"train кадры {args.train_start}..{args.train_end}, "
          f"holdout {args.holdout_start}..{args.holdout_end}")
    for name, s in per_cam.items():
        if not s["train"]["n"]:
            print(f"  {name}: нет наблюдений в train — сдвиг не оценивается")
            continue
        off = s["offset_px"]
        print(f"  {name}: train n={s['train']['n']} mean={s['train']['mean_px']} px, "
              f"offset={off} px -> holdout "
              f"{s['holdout_before']['mean_px'] if s['holdout_before']['n'] else '—'} px -> "
              f"{s['holdout_after']['mean_px'] if s['holdout_after']['n'] else '—'} px "
              f"(n={s['holdout_after']['n']})")

    doc = {
        "method": "median(bbox_center - project(pivot_gt)) по камерам на train-кадрах",
        "offsets_px": offsets,
        "camera_names": ds.names,
        "train": {"frames": [args.train_start, args.train_end]},
        "holdout": {"frames": [args.holdout_start, args.holdout_end]},
        "per_camera": per_cam,
        "note": ("Сдвиг — приближение: смещение bbox↔pivot зависит от ракурса и "
                 "дистанции. Оценён на train, на holdout только проверен."),
    }
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"записано: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
