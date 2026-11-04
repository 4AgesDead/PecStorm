"""Свип параметров геометрии/слияния по готовым детекциям: MAE vs coverage.

Запуск:
    python experiments/geometry_sweep.py --detections out/truck/detections.csv
        [--start 0 --end 300 --eval-step 5] [--out out/experiments]

Для каждой комбинации ``max_reprojection_px``, ``min_views``, ``gate``:
* реконструкция на подмножестве кадров (тот же вход, меняются только пороги);
* MAE/RMSE/P95 по MEASURED против GT и покрытие — отдельно (§7.6 брифа).

Главный смысл: показать компромисс «принять больше измерений ↔ выше ошибка»,
а не подобрать «лучшее» число под GT. Результаты — диагностика.

Отчёт: ``out/experiments/geometry_sweep.json`` + печать таблицы.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from multiview3d.fusion import FusionConfig
from multiview3d.metrics import evaluate_truck
from multiview3d.truck_dataset import TruckDataset
from multiview3d.truck_pipeline import (
    estimate_points,
    load_anchor_offset,
    load_anchor_offset_views,
    load_anchor_px_offset,
    load_detections,
    run_reconstruction,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Свип порогов слияния на детекциях «Пеленги»")
    ap.add_argument("--root", default="scene1/zadania")
    ap.add_argument("--detections", default="out/truck/detections.csv")
    ap.add_argument("--anchor-offset", default="datasets/truck/anchor_offset.json")
    ap.add_argument("--anchor-px-offset", default="datasets/truck/anchor_px_offset.json")
    ap.add_argument("--anchor-offset-views", default="datasets/truck/anchor_offset_views.json")
    ap.add_argument("--no-anchor-px-offset", action="store_true")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=300)
    ap.add_argument("--eval-step", type=int, default=5)
    ap.add_argument("--max-reproj", default="10,20,30,50")
    ap.add_argument("--min-views", default="2,3")
    ap.add_argument("--gates", default="3,5")
    ap.add_argument("--out", default="out/experiments")
    args = ap.parse_args(argv)

    ds = TruckDataset.discover(args.root)
    dets = load_detections(args.detections)
    frames = [f for f in range(args.start, args.end + 1, args.eval_step) if f in dets]
    if not frames:
        print("нет детекций в выбранном диапазоне", file=sys.stderr)
        return 2
    gt = ds.ground_truth_by_frame()
    offset = load_anchor_offset(args.anchor_offset)
    px_offset = None if args.no_anchor_px_offset else load_anchor_px_offset(args.anchor_px_offset)
    views_offset = load_anchor_offset_views(args.anchor_offset_views)

    runs = []
    print(f"кадры {frames[0]}..{frames[-1]} шаг {args.eval_step} ({len(frames)} кадров), "
          f"сдвиг якоря: {offset}, 2D-сдвиг по камерам: "
          f"{'нет' if not px_offset else px_offset}, "
          f"наборов видов: {'нет' if not views_offset else len(views_offset)}")
    print(f"{'max_reproj':>10} {'min_views':>9} {'gate':>5} {'measured':>9} "
          f"{'MAE мм':>9} {'RMSE':>9} {'P95':>9} {'покрытие':>10}")
    for mr in [float(x) for x in args.max_reproj.split(",")]:
        for mv in [int(x) for x in args.min_views.split(",")]:
            for gate in [float(x) for x in args.gates.split(",")]:
                cfg = FusionConfig(association_mode="single_target", min_views=mv,
                                   max_reprojection_px=mr, gate_mahalanobis=gate)
                res = run_reconstruction(ds, dets, fusion=cfg, anchor_offset_m=offset,
                                         anchor_px_offset=px_offset,
                                         anchor_offset_views=views_offset,
                                         frames=frames)
                rep = evaluate_truck(estimate_points(res), gt, frames=frames, fps=ds.fps)
                doc = rep.to_dict()
                m = doc["measured"]
                cov = doc["coverage"]
                runs.append({"max_reprojection_px": mr, "min_views": mv,
                             "gate_mahalanobis": gate, **doc})
                print(f"{mr:>10.0f} {mv:>9d} {gate:>5.0f} {m['n_points']:>9d} "
                      f"{(m['mae_mm'] if m['mae_mm'] is not None else float('nan')):>9.1f} "
                      f"{(m['rmse_mm'] if m['rmse_mm'] is not None else float('nan')):>9.1f} "
                      f"{(m['p95_mm'] if m['p95_mm'] is not None else float('nan')):>9.1f} "
                      f"{cov['measured']:>10.3f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    doc = {"detections_csv": str(args.detections).replace("\\", "/"),
           "frames": [frames[0], frames[-1], args.eval_step],
           "anchor_offset_m": offset,
           "anchor_px_offset": px_offset,
           "anchor_offset_views": views_offset,
           "note": ("Свип показывает компромисс порогов на одних и тех же "
                    "детекциях; это диагностика, а не подбор под hidden test."),
           "runs": runs}
    (out / "geometry_sweep.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"JSON: {out / 'geometry_sweep.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
