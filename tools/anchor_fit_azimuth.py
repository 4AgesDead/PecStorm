"""Фит якоря, зависящего от ракурса (азимут камеры относительно курса).

Схема:
1. Реконструкция **без** азимутальной поправки (только константы по наборам видов).
2. Для каждой MEASURED-точки считается невязка `estimate - gt` и азимут
   `bearing(камера→объект) − heading(объект)`; курс — из GT-скорости.
3. По каждому набору видов (n ≥ min-observations) фитится ряд Фурье до 2-й
   гармоники.
4. Честные проверки: holdout 700..1000 и parity (чётные → нечётные кадры).

Запуск:
    python tools/anchor_fit_azimuth.py --detections out/truck/detections_ens.csv `
        --anchor-views out/truck/anchor_ens.json `
        --out datasets/truck/anchor_azimuth.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from multiview3d.anchor_azimuth import basis, fit_coefficients, save_coefficients, wrap_angle
from multiview3d.fusion import FusionConfig
from multiview3d.truck_dataset import DEFAULT_REPAIRS, DEFAULT_ROOT, TruckDataset
from multiview3d.truck_pipeline import (
    estimate_points,
    load_anchor_offset_views,
    load_detections,
    run_reconstruction,
)
from tools.fit_truck_model import heading_from_gt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Фит азимутального якоря")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--repairs", default=str(DEFAULT_REPAIRS))
    ap.add_argument("--detections", required=True)
    ap.add_argument("--anchor-views", default="out/truck/anchor_ens.json")
    ap.add_argument("--max-reprojection-px", type=float, default=50.0)
    ap.add_argument("--gate", type=float, default=10.0)
    ap.add_argument("--fit-end", type=int, default=700)
    ap.add_argument("--min-observations", type=int, default=40)
    ap.add_argument("--out", default="datasets/truck/anchor_azimuth.json")
    args = ap.parse_args(argv)

    ds = TruckDataset.discover(args.root, repairs_path=args.repairs or None)
    gt = ds.ground_truth_by_frame()
    yaw_gt = heading_from_gt(gt)
    dets = load_detections(args.detections)
    anchors = load_anchor_offset_views(args.anchor_views)
    cfg = FusionConfig(association_mode="single_target",
                       max_reprojection_px=args.max_reprojection_px,
                       gate_mahalanobis=args.gate)
    res = run_reconstruction(ds, dets, fusion=cfg, anchor_offset_views=anchors)

    from multiview3d.anchor_azimuth import heading_from_positions

    samples = []
    for track in res.pipeline.all_tracks():
        points = sorted(track.points, key=lambda p: p.timestamp)
        for i, p in enumerate(points):
            if p.frame is None or p.frame not in gt or p.status != "MEASURED":
                continue
            views = tuple(sorted(p.views or ()))
            if len(views) < 2:
                continue
            heading = heading_from_positions(points, i)   # как на рантайме
            if heading is None:
                continue
            cams = ds.provider.calibrations_at(p.frame, nearest=True)
            cam = cams.get(views[0])
            if cam is None:
                continue
            C = -np.asarray(cam.R, dtype=float).T @ np.asarray(cam.t, dtype=float).reshape(3)
            P = np.asarray(p.position, dtype=float)
            bearing = float(np.arctan2(P[1] - C[1], P[0] - C[0]))
            az = wrap_angle(bearing - heading)
            samples.append({"frame": p.frame, "views": views, "az": az,
                            "residual": P - np.asarray(gt[p.frame], dtype=float)})
    print(f"наблюдений: {len(samples)} (курс — по точкам трека, как на рантайме)")

    train = [s for s in samples if s["frame"] <= args.fit_end]
    test = [s for s in samples if s["frame"] > args.fit_end]
    even = [s for s in samples if s["frame"] % 2 == 0]
    odd = [s for s in samples if s["frame"] % 2 == 1]

    def mae(items, coefs):
        errs = []
        for s in items:
            key = "|".join(s["views"])
            coef = coefs.get(key)
            corr = (basis(s["az"]) @ np.asarray(coef)).reshape(3) if coef is not None else 0.0
            errs.append(float(np.linalg.norm(s["residual"] - corr)))
        return float(np.mean(errs)) if errs else float("nan")

    coefs = fit_coefficients(train, min_observations=args.min_observations)
    print(f"наборов видов с моделью: {len(coefs)}")
    print(f"train:   {mae(train, {}):.2f} → {mae(train, coefs):.2f} м")
    print(f"holdout: {mae(test, {}):.2f} → {mae(test, coefs):.2f} м "
          f"(n={len(test)})")

    coefs_even = fit_coefficients(even, min_observations=args.min_observations)
    coefs_odd = fit_coefficients(odd, min_observations=args.min_observations)
    print(f"parity:  фит чётные → нечётные {mae(odd, coefs_even):.2f} м; "
          f"фит нечётные → чётные {mae(even, coefs_odd):.2f} м")

    save_coefficients(args.out, coefs, {
        "method": "МНК по ряду Фурье (2 гармоники) на остатках estimate-gt по набору видов",
        "fit_frames": [0, args.fit_end],
        "holdout_frames": [args.fit_end + 1, 1000],
        "detections_csv": str(args.detections).replace("\\", "/"),
        "anchor_views": str(args.anchor_views).replace("\\", "/"),
        "note": ("Курс в фите — по GT-скорости; на рантайме — по соседним точкам трека. "
                 "Поправка вычитается из оценки."),
    })
    print(f"записано: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
