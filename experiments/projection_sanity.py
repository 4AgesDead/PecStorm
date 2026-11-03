"""Проверка согласованности проекции: GT-пивот против 2D-детекций по камерам.

Запуск:
    python experiments/projection_sanity.py --detections out/truck/detections.csv
        [--step 10] [--out out/experiments]

Что измеряется: расстояние (px) между проекцией GT-пивота и центром bbox
детектора на тех же кадрах. Это **не** оценка точности конвейера, а проверка
геометрии (оси, единицы, intrinsics, позы). Большие значения означают, что
2D-якорь детектора (центр bbox) далеко от pivot актора, и/или калибровка
смещена — см. docs/OPEN_QUESTIONS.md и docs/TRUCK_DATASET.md.

Отчёт: ``projection_sanity.json`` (по камерам: n, mean/median/p90/max px,
доля проекций вне кадра).
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GT-пивот vs 2D-детекции: согласованность проекции")
    ap.add_argument("--root", default="huniashahatona/zadania")
    ap.add_argument("--detections", default="out/truck/detections.csv")
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--out", default="out/experiments")
    args = ap.parse_args(argv)

    ds = TruckDataset.discover(args.root)
    gt = ds.ground_truth_by_frame()
    dets = load_detections(args.detections)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dist: dict[str, list[float]] = {n: [] for n in ds.names}
    offscreen: dict[str, int] = {n: 0 for n in ds.names}
    for frame in sorted(dets):
        if frame % args.step or frame not in gt:
            continue
        cams = ds.provider.calibrations_at(frame, nearest=True)
        for name, dets_cam in dets[frame].items():
            if not dets_cam:
                continue
            best = max(dets_cam, key=lambda d: d.score)
            cam = cams.get(name)
            if cam is None:
                continue
            try:
                uv = project_point(cam, gt[frame])
            except ValueError:
                offscreen[name] += 1
                continue
            dist[name].append(float(np.linalg.norm(np.asarray(best.center) - uv)))

    def stats(values: list[float]) -> dict:
        if not values:
            return {"n": 0}
        a = np.asarray(values)
        return {"n": int(a.size), "mean_px": round(float(a.mean()), 2),
                "median_px": round(float(np.median(a)), 2),
                "p90_px": round(float(np.percentile(a, 90)), 2),
                "max_px": round(float(a.max()), 2)}

    doc = {
        "detections_csv": str(args.detections).replace("\\", "/"),
        "step": args.step,
        "metric": "||bbox_center - project(pivot_gt)||, px",
        "note": ("Проверка трансформации координат, а не точности трекера: "
                 "bbox center != actor pivot по определению."),
        "per_camera": {n: {**stats(dist[n]), "offscreen_projections": offscreen[n]}
                       for n in ds.names},
    }
    (out_dir / "projection_sanity.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")

    print("Расстояние bbox-центр <-> проекция GT-пивота (px):")
    allv = []
    for name in ds.names:
        s = doc["per_camera"][name]
        allv.extend(dist[name])
        if s["n"]:
            print(f"  {name}: n={s['n']}, mean={s['mean_px']}, median={s['median_px']}, "
                  f"p90={s['p90_px']}, max={s['max_px']}, вне кадра {s['offscreen_projections']}")
        else:
            print(f"  {name}: нет детекций на выбранных кадрах")
    if allv:
        a = np.asarray(allv)
        print(f"Всего: n={a.size}, mean={a.mean():.2f} px, median={np.median(a):.2f} px")
    print(f"JSON: {out_dir / 'projection_sanity.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
