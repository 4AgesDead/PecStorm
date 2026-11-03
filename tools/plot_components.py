"""График «Итоговый результат»: X/Y/Z от времени, эталон vs оценка.

Стиль как в презентации организаторов: сплошные линии — эталон (`X_t`, `Y_t`,
`Z_t`), штриховые с точками — оценка по детекциям (`X_det`, `Y_det`, `Z_det`),
плюс отдельная панель «расхождение с эталоном» (оценка − эталон) по осям.

Источники: GT — `object_transform_SM_AA_track.csv` через `TruckDataset`
(битые кадры исключаются так же, как в оценке), оценки — CSV/JSON из
`apps/reconstruct_truck.py`.

Запуск:
    python tools/plot_components.py --estimates out/truck/estimates.csv `
        --out out/truck/report/components_vs_time.png `
        --error-out out/truck/report/components_error_vs_time.png

Не досчитывает траекторию и не сглаживает: рисует ровно то, что есть в файле
оценок. Кадры без оценки остаются разрывами в штриховых линиях.
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

from multiview3d.truck_dataset import DEFAULT_REPAIRS, DEFAULT_ROOT, TruckDataset
from multiview3d.truck_pipeline import load_estimates
from multiview3d.types import MEASURED, PREDICTED_ONLY
from multiview3d.visualize import plot_components_vs_time


def build_series(gt: dict, points, frames: list[int]):
    """Массивы (N, 3) для эталона, измерений и прогнозов; NaN — нет данных."""
    n = len(frames)
    gt_xyz = np.full((n, 3), np.nan)
    meas = np.full((n, 3), np.nan)
    pred = np.full((n, 3), np.nan)
    index = {f: i for i, f in enumerate(frames)}

    for f, p in gt.items():
        if f in index:
            gt_xyz[index[f]] = np.asarray(p, dtype=float)

    # При нескольких точках на кадр предпочитаем MEASURED и максимальную confidence.
    best: dict[int, tuple[int, float, object]] = {}
    for p in points:
        if p.frame is None or p.frame not in index:
            continue
        rank = 2 if p.status == MEASURED else (1 if p.status == PREDICTED_ONLY else 0)
        conf = p.confidence if p.confidence is not None else 0.0
        cur = best.get(p.frame)
        if cur is None or (rank, conf) > (cur[0], cur[1]):
            best[p.frame] = (rank, conf, p)
    for f, (rank, _, p) in best.items():
        target = meas if rank == 2 else pred
        target[index[f]] = np.asarray(p.position, dtype=float)
    return gt_xyz, meas, pred


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="График X/Y/Z: эталон vs оценка")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--repairs", default=str(DEFAULT_REPAIRS))
    ap.add_argument("--estimates", default="out/truck/estimates.csv")
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--no-predicted", action="store_true",
                    help="не рисовать прогноз Калмана (только 6 линий: X/Y/Z_t и X/Y/Z_det)")
    ap.add_argument("--out", default="out/truck/report/components_vs_time.png")
    ap.add_argument("--error-out", default="out/truck/report/components_error_vs_time.png")
    ap.add_argument("--path-out", default="out/truck/report/components_with_path.png",
                    help="фигура «компоненты + траектория в плане»; пусто — не писать")
    ap.add_argument("--json", default="out/truck/report/components_vs_time.json")
    args = ap.parse_args(argv)

    ds = TruckDataset.discover(args.root, repairs_path=args.repairs or None)
    gt = ds.ground_truth_by_frame()
    points = load_estimates(args.estimates)
    if not points:
        print(f"в {args.estimates} нет оценок", file=sys.stderr)
        return 2

    est_frames = {p.frame for p in points if p.frame is not None}
    frames = sorted(set(gt) | est_frames)
    if args.start is not None:
        frames = [f for f in frames if f >= args.start]
    if args.end is not None:
        frames = [f for f in frames if f <= args.end]
    if not frames:
        print("нет кадров в выбранном диапазоне", file=sys.stderr)
        return 2

    gt_xyz, meas, pred = build_series(gt, points, frames)
    n_pred_rows = int(np.isfinite(pred).all(axis=1).sum())
    if args.no_predicted:
        pred = None
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    err_out = Path(args.error_out) if args.error_out else None
    if err_out is not None:
        err_out.parent.mkdir(parents=True, exist_ok=True)

    stats = plot_components_vs_time(
        frames, gt_xyz, meas, pred, path=out, error_path=err_out,
        path_panel_path=(Path(args.path_out) if args.path_out else None),
        title=("Итоговый результат: компоненты положения — "
               "эталон (—) и оценка по детекциям (---)"),
        fps=ds.fps)

    doc = {
        "estimates_csv": str(args.estimates).replace("\\", "/"),
        "frames": [frames[0], frames[-1]],
        "fps": ds.fps,
        "measured_rows": int(np.isfinite(meas).all(axis=1).sum()),
        "predicted_rows": n_pred_rows,
        "gt_rows": int(np.isfinite(gt_xyz).all(axis=1).sum()),
        "stats": stats,
        "note": ("Значения — в метрах; ось времени в секундах (кадр = t · fps). "
                 "Разрывы штриховых линий — кадры без оценки. Расхождения "
                 "считаны только там, где есть и эталон, и измерение."),
        "png": str(out).replace("\\", "/"),
        "png_error": None if err_out is None else str(err_out).replace("\\", "/"),
        "png_with_path": (str(Path(args.path_out)).replace("\\", "/")
                          if args.path_out else None),
    }
    pj = Path(args.json)
    pj.parent.mkdir(parents=True, exist_ok=True)
    pj.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"кадры {frames[0]}..{frames[-1]} ({len(frames)}), "
          f"GT {doc['gt_rows']}, MEASURED {doc['measured_rows']}, "
          f"PREDICTED_ONLY {doc['predicted_rows']}")
    if stats.get("mae_m"):
        m = stats["mae_m"]
        print(f"MAE по осям: X={m['X']} м, Y={m['Y']} м, Z={m['Z']} м; "
              f"MAE={stats['mae_total_m']} м (n={stats['n']})")
    print(f"PNG: {out}" + (f", {err_out}" if err_out else "")
          + (f", {args.path_out}" if args.path_out else "") + f"; JSON: {pj}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
