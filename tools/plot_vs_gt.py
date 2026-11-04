"""График сравнения оценки с эталоном (GT): траектория X–Y и ошибка по времени.

Источники:
* GT — полный эталон датасета через ``TruckDataset`` (битые кадры исключены);
* оценка и ошибка — пофреймовый CSV из ``apps/evaluate_truck.py``
  (``out/truck/eval_final/evalpf.csv``);
* покрытие — JSON оценки (``eval.json``), если он есть.

Запуск:
    py tools/plot_vs_gt.py `
        --per-frame out/truck/eval_final/evalpf.csv `
        --eval out/truck/eval_final/eval.json `
        --out results/truck/report_final/comparison_vs_gt.png
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import rcParams  # noqa: E402

from multiview3d.truck_dataset import DEFAULT_REPAIRS, DEFAULT_ROOT, TruckDataset  # noqa: E402

rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
                 "figure.facecolor": "white", "axes.facecolor": "white"})


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _num(row: dict, key: str):
    v = row.get(key)
    if v in (None, "", "None", "nan"):
        return None
    try:
        x = float(v)
    except ValueError:
        return None
    return x if np.isfinite(x) else None


def stats(errors: list[float]) -> dict:
    if not errors:
        return {"n": 0, "mae_m": None, "median_m": None, "p95_m": None}
    a = np.asarray(errors, dtype=float)
    return {"n": int(a.size), "mae_m": float(a.mean()),
            "median_m": float(np.median(a)), "p95_m": float(np.percentile(a, 95))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="График «оценка vs эталон»")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--repairs", default=str(DEFAULT_REPAIRS))
    ap.add_argument("--per-frame", default="out/truck/eval_final/evalpf.csv")
    ap.add_argument("--eval", default="out/truck/eval_final/eval.json")
    ap.add_argument("--out", default="results/truck/report_final/comparison_vs_gt.png")
    ap.add_argument("--json-out", default=None,
                    help="JSON с числами (по умолчанию рядом с PNG)")
    ap.add_argument("--holdout-start", type=int, default=700)
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--title", default="Сравнение с эталоном: финальная конфигурация "
                                      "v1 + v2 + v4b, якоря + азимут")
    args = ap.parse_args(argv)

    per_frame = Path(args.per_frame)
    if not per_frame.exists():
        print(f"нет файла {per_frame}", file=sys.stderr)
        return 2
    rows = read_rows(per_frame)
    ds = TruckDataset.discover(args.root, repairs_path=args.repairs or None)
    fps = float(args.fps or ds.fps)
    gt = ds.ground_truth_by_frame()
    lo, hi = ds.common_frame_range()
    gt_frames = [f for f in sorted(gt) if lo <= f <= hi]

    meas_xy, meas_err = [], []
    meas_frames: list[int] = []
    pred_xy, pred_err = [], []
    for r in rows:
        x, y, e = _num(r, "x"), _num(r, "y"), _num(r, "error_mm")
        if x is None or y is None:
            continue
        status = (r.get("status") or "").strip()
        if status == "MEASURED":
            if e is not None:
                meas_xy.append((x, y))
                meas_err.append(e / 1000.0)
                meas_frames.append(int(r["frame"]))
        elif status == "PREDICTED_ONLY":
            pred_xy.append((x, y))
            if e is not None:
                pred_err.append(e / 1000.0)

    gt_xy = np.array([[gt[f][0], gt[f][1]] for f in gt_frames], dtype=float)

    eval_doc = {}
    if args.eval and Path(args.eval).exists():
        eval_doc = json.loads(Path(args.eval).read_text(encoding="utf-8"))
    coverage = float(((eval_doc.get("coverage") or {}).get("measured")) or np.nan)
    if not np.isfinite(coverage):
        coverage = len(meas_xy) / max(len(gt_frames), 1)

    all_stats = stats(meas_err)
    if all_stats["n"] == 0:
        print("в CSV нет измеренных точек с ошибкой", file=sys.stderr)
        return 2
    hold_frames = [f for f in gt_frames if f >= args.holdout_start]
    hold_errors = []
    hold_measured = 0
    for r in rows:
        f = _num(r, "frame")
        if f is None or f < args.holdout_start:
            continue
        if (r.get("status") or "") == "MEASURED":
            e = _num(r, "error_mm")
            if e is not None:
                hold_errors.append(e / 1000.0)
                hold_measured += 1
    hold_stats = stats(hold_errors)
    hold_coverage = hold_measured / max(len(hold_frames), 1)

    # ---- фигура ----
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14.5, 6.6), gridspec_kw={"width_ratios": [1.0, 1.05]})
    fig.suptitle(args.title, fontsize=13, fontweight="bold")

    ax1.plot(gt_xy[:, 0], gt_xy[:, 1], color="black", lw=2.0, label="эталон (GT)")
    # путь оценки — тонкая линия только внутри непрерывных участков
    seg: list[tuple[float, float]] = []
    prev_f = None
    for f, xy in zip(meas_frames, meas_xy):
        if prev_f is not None and f == prev_f + 1:
            seg.append(xy)
        else:
            if len(seg) >= 2:
                a = np.asarray(seg)
                ax1.plot(a[:, 0], a[:, 1], color="#1b7f3b", lw=1.2, alpha=0.85,
                         zorder=2)
            seg = [xy]
        prev_f = f
    if len(seg) >= 2:
        a = np.asarray(seg)
        ax1.plot(a[:, 0], a[:, 1], color="#1b7f3b", lw=1.2, alpha=0.85,
                 zorder=2, label="путь оценки")
    if meas_xy:
        m = np.asarray(meas_xy)
        sc = ax1.scatter(m[:, 0], m[:, 1], c=np.asarray(meas_err), cmap="viridis",
                         vmin=0.0, vmax=float(np.percentile(meas_err, 95)),
                         s=14, linewidths=0, label="оценка (MEASURED)",
                         zorder=3)
        cb = fig.colorbar(sc, ax=ax1, fraction=0.046, pad=0.03)
        cb.set_label("ошибка против GT, м")
    if pred_xy:
        p = np.asarray(pred_xy)
        ax1.scatter(p[:, 0], p[:, 1], marker="x", s=26, color="#d17b00",
                    label="прогноз (PREDICTED_ONLY)", zorder=3)
    ax1.scatter(gt_xy[0, 0], gt_xy[0, 1], s=60, facecolors="none",
                edgecolors="#1b7f3b", linewidths=2, zorder=4, label="старт GT")
    ax1.scatter(gt_xy[-1, 0], gt_xy[-1, 1], s=60, marker="s", facecolors="none",
                edgecolors="#a03030", linewidths=2, zorder=4, label="финиш GT")
    ax1.set_xlabel("X, м")
    ax1.set_ylabel("Y, м")
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.set_title("Траектория в плане (мировые координаты)")
    ax1.legend(loc="best", fontsize=8, framealpha=0.9)

    t_meas = np.array([int(r["frame"]) / fps for r in rows
                       if (r.get("status") or "") == "MEASURED"
                       and _num(r, "error_mm") is not None], dtype=float)
    ax2.axvspan(args.holdout_start / fps, gt_frames[-1] / fps, color="#f0f0f0",
                zorder=0, label=f"holdout {args.holdout_start}–{gt_frames[-1]}")
    if len(t_meas) and meas_err:
        ax2.scatter(t_meas, np.asarray(meas_err), s=10, color="#1b7f3b",
                    alpha=0.75, label="MEASURED", zorder=2)
    if pred_err:
        t_pred = np.array([int(r["frame"]) / fps for r in rows
                           if (r.get("status") or "") == "PREDICTED_ONLY"
                           and _num(r, "error_mm") is not None], dtype=float)
        ax2.scatter(t_pred, np.asarray(pred_err), marker="x", s=30,
                    color="#d17b00", label="PREDICTED_ONLY", zorder=2)
    mae = all_stats["mae_m"]
    if mae is not None:
        ax2.axhline(mae, color="#a03030", ls="--", lw=1.4,
                    label=f"MAE = {mae:.2f} м")
    ax2.set_xlabel("время, с")
    ax2.set_ylabel("ошибка 3D против GT, м")
    ax2.set_title("Ошибка по кадрам")
    ax2.set_xlim(0, gt_frames[-1] / fps)
    ax2.legend(loc="upper left", fontsize=8, framealpha=0.9)

    box = (f"n = {all_stats['n']}\n"
           f"MAE = {all_stats['mae_m']:.2f} м\n"
           f"медиана = {all_stats['median_m']:.2f} м\n"
           f"P95 = {all_stats['p95_m']:.2f} м\n"
           f"покрытие = {coverage * 100:.1f} %\n"
           f"holdout: {hold_stats['mae_m']:.2f} м / {hold_coverage * 100:.1f} %")
    ax2.text(0.99, 0.97, box, transform=ax2.transAxes, ha="right", va="top",
             fontsize=9, family="monospace",
             bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                       edgecolor="#999999", alpha=0.92))

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)

    doc = {
        "per_frame_csv": str(per_frame).replace("\\", "/"),
        "eval_json": str(args.eval).replace("\\", "/") if args.eval else None,
        "gt_frames": [gt_frames[0], gt_frames[-1]],
        "fps": fps,
        "measured": all_stats,
        "predicted": stats(pred_err),
        "coverage_measured": coverage,
        "holdout": {"start": args.holdout_start, **hold_stats,
                    "coverage_measured": hold_coverage},
        "png": str(out).replace("\\", "/"),
    }
    json_out = Path(args.json_out) if args.json_out else out.with_suffix(".json")
    json_out.write_text(json.dumps(doc, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"MEASURED: n={all_stats['n']}, MAE={all_stats['mae_m']:.3f} м, "
          f"медиана={all_stats['median_m']:.3f}, P95={all_stats['p95_m']:.3f}, "
          f"покрытие={coverage * 100:.1f} %")
    print(f"holdout {args.holdout_start}..{gt_frames[-1]}: "
          f"MAE={hold_stats['mae_m']:.3f} м, покрытие={hold_coverage * 100:.1f} %")
    print(f"график: {out}")
    print(f"числа:  {json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
