"""Оценка 3D против GT датасета «Пеленга»: ошибка отдельно, покрытие отдельно.

Запуск:
    python apps/evaluate_truck.py --estimates out/truck/estimates.csv
        --out-json out/truck/evaluation.json --out-csv out/truck/eval_per_frame.csv

Метрики (§7.6 брифа):
* MAE/RMSE/медиана/P95/P99/max по **измерениям** (``MEASURED``), мм;
* отдельно — статистика по прогнозам (``PREDICTED_ONLY``): это не точность
  измерения, а качество экстраполяции;
* покрытие: доля кадров с измерением / только прогнозом / без оценки;
* подмножества по набору видов (``cam1|cam3`` и т.п.).

GT берётся из CSV организаторов; битые ячейки (кадры 402 и 691) в GT не
попадают, поэтому в отчёте перечислены ``gt_invalid_frames``. Sidecar-репары
по умолчанию не применяются (``verified_by_organizer=false``); включить можно
флагом ``--accept-unverified-repairs``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from multiview3d.metrics import evaluate_truck, per_frame_csv_rows
from multiview3d.truck_dataset import DEFAULT_REPAIRS, DEFAULT_ROOT, TruckDataset
from multiview3d.truck_pipeline import load_estimates


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Оценка 3D по GT датасета «Пеленга»")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--repairs", default=str(DEFAULT_REPAIRS))
    ap.add_argument("--estimates", required=True, help="CSV из apps/reconstruct_truck.py")
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--accept-unverified-repairs", action="store_true",
                    help="применить sidecar-репары GT (organizer не подтверждал)")
    ap.add_argument("--out-json", default="out/truck/evaluation.json")
    ap.add_argument("--out-csv", default="out/truck/eval_per_frame.csv")
    args = ap.parse_args(argv)

    ds = TruckDataset.discover(args.root, repairs_path=args.repairs or None)
    pts = load_estimates(args.estimates)
    if not pts:
        print(f"в {args.estimates} нет оценок", file=sys.stderr)
        return 2

    gt = ds.ground_truth(accept_unverified=args.accept_unverified_repairs)
    gt_by_frame = {int(f): p for f, p in zip(gt.frames, gt.positions_m)}
    lo, hi = ds.common_frame_range()
    start = lo if args.start is None else int(args.start)
    end = hi if args.end is None else int(args.end)
    frames = list(range(start, end + 1))

    rep = evaluate_truck(pts, gt_by_frame, frames=frames, fps=ds.fps, unit=1.0,
                         gt_invalid_frames=sorted({c["frame"] for c in gt.invalid_cells}))
    doc = {
        "dataset": "pelenga_truck",
        "estimates_csv": str(args.estimates).replace("\\", "/"),
        "frame_range": [start, end],
        "fps": ds.fps,
        "repairs_applied": bool(args.accept_unverified_repairs),
        **rep.to_dict(),
    }
    p = Path(args.out_json)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = per_frame_csv_rows(rep)
    pc = Path(args.out_csv)
    pc.parent.mkdir(parents=True, exist_ok=True)
    with pc.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["frame"])
        w.writeheader()
        w.writerows(rows)

    m = doc["measured"]
    c = doc["coverage"]
    print(f"Оценка по кадрам {start}..{end} (fps={ds.fps:g}, единицы — мм)")
    print(f"Покрытие: измерено {c['measured']:.1%}, только прогноз "
          f"{c['progress_predicted_only']:.1%}, без оценки {c['unobservable']:.1%}, "
          f"один вид {c['monocular_only']:.1%}")
    if m["n_points"]:
        print(f"Ошибка MEASURED (n={m['n_points']}): MAE={m['mae_mm']:.1f} мм, "
              f"RMSE={m['rmse_mm']:.1f}, медиана={m['median_mm']:.1f}, "
              f"P95={m['p95_mm']:.1f}, P99={m['p99_mm']:.1f}, max={m['max_mm']:.1f}")
        if m["per_axis_mae_mm"]:
            ax = m["per_axis_mae_mm"]
            print(f"  MAE по осям: X={ax[0]:.1f} Y={ax[1]:.1f} Z={ax[2]:.1f} мм")
    else:
        print("Ошибка MEASURED: нет точек с измерениями 3D")
    pr = doc["predicted"]
    if pr["n_points"]:
        print(f"Прогнозы PREDICTED_ONLY (n={pr['n_points']}, не точность измерения): "
              f"MAE={pr['mae_mm']:.1f} мм, P95={pr['p95_mm']:.1f}, max={pr['max_mm']:.1f}")
    for key, sub in doc["per_camera_subset"].items():
        print(f"  виды {key}: n={sub['n_points']}, MAE={sub['mae_mm']} мм")
    if doc["gt_invalid_frames"]:
        print(f"Кадры без GT (битые ячейки, не оценены): {doc['gt_invalid_frames']}")
    print(f"Отчёт: {args.out_json}; по кадрам: {args.out_csv} ({len(rows)} строк)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
