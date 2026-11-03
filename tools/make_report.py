"""Сводный отчёт по truck-workflow: JSON-сводка и графики.

Запуск:
    python tools/make_report.py [--eval out/truck/evaluation.json]
        [--per-frame out/truck/eval_per_frame.csv]
        [--estimates out/truck/estimates.csv] [--out out/truck/report]

Что делает:
* собирает в один JSON факты из артефактов: статистику детекции, сдвиг якоря,
  видимость, оценку (ошибка + покрытие), бенчмарк детектора, робастность;
* строит графики: ошибка по кадрам, траектория XY (GT vs оценка),
  покрытие по статусам.

Ничего не досчитывает и не «улучшает»: только читает файлы, которых может не
быть (тогда раздел помечается как отсутствующий).
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


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_error_vs_frame(rows: list[dict], out: Path) -> bool:
    pts = [r for r in rows if r.get("error_mm") not in (None, "")]
    if not pts:
        return False
    x = np.array([int(r["frame"]) for r in pts])
    y = np.array([float(r["error_mm"]) for r in pts]) / 1000.0  # мм -> м
    status = [r.get("status") or "" for r in pts]
    fig, ax = plt.subplots(figsize=(10, 4))
    for label, color in (("MEASURED", "#1b7f3b"), ("PREDICTED_ONLY", "#d17b00")):
        m = np.array([s == label for s in status])
        if m.any():
            ax.scatter(x[m], y[m], s=10, color=color, label=label)
    ax.set_xlabel("кадр")
    ax.set_ylabel("ошибка 3D против GT, м")
    ax.set_title("Ошибка оценки по кадрам (GT-кадры)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return True


def plot_trajectory(rows: list[dict], out: Path) -> bool:
    rows = [r for r in rows if r.get("gt_x") not in (None, "")]
    if not rows:
        return False
    gt = np.array([[float(r["gt_x"]), float(r["gt_y"])] for r in rows])
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(gt[:, 0], gt[:, 1], color="black", lw=1.2, label="GT (pivot)")
    for label, color in (("MEASURED", "#1b7f3b"), ("PREDICTED_ONLY", "#d17b00")):
        pts = [r for r in rows if (r.get("status") or "") == label
               and r.get("x") not in (None, "")]
        if pts:
            arr = np.array([[float(r["x"]), float(r["y"])] for r in pts])
            ax.scatter(arr[:, 0], arr[:, 1], s=12, color=color, label=label)
    ax.set_xlabel("X, м")
    ax.set_ylabel("Y, м")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("Траектория в плане (мировая система, метры)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return True


def plot_components(rows: list[dict], gt_by_frame: dict | None, out: Path,
                    err_out: Path, fps: float = 30.0,
                    with_predicted: bool = True,
                    path_out: Path | None = None) -> bool:
    """X/Y/Z от времени: эталон (сплошные) и оценка (штриховые) + ошибка по осям.

    Эталон берётся из ``gt_by_frame`` (полный GT датасета): в CSV оценки GT
    заполнен только на кадрах с оценкой, поэтому график «только по CSV» рвётся
    на неоценённых кадрах. Если словарь не передан, используется GT из строк.
    """
    rows = [r for r in rows if r.get("frame") not in (None, "")]
    if not rows:
        return False
    from multiview3d.visualize import plot_components_vs_time

    frames = sorted(set(gt_by_frame or {}) | {int(r["frame"]) for r in rows})
    index = {f: i for i, f in enumerate(frames)}
    n = len(frames)
    gt = np.full((n, 3), np.nan)
    meas = np.full((n, 3), np.nan)
    pred = np.full((n, 3), np.nan)

    def triple(r):
        keys = ("x", "y", "z")
        if any(r.get(k) in (None, "") for k in keys):
            return None
        return [float(r[k]) for k in keys]

    if gt_by_frame:
        for f, p in gt_by_frame.items():
            if f in index:
                gt[index[f]] = np.asarray(p, dtype=float)

    for r in sorted(rows, key=lambda x: (x.get("status") != "MEASURED",
                                         x.get("status") != "PREDICTED_ONLY")):
        i = index[int(r["frame"])]
        if not gt_by_frame and r.get("gt_x") not in (None, ""):
            gt[i] = [float(r["gt_x"]), float(r["gt_y"]), float(r["gt_z"])]
        est = triple(r)
        status = r.get("status") or ""
        if est is not None and status == "MEASURED":
            meas[i] = est
        elif est is not None and status == "PREDICTED_ONLY":
            pred[i] = est

    if not np.isfinite(gt).any():
        return False
    plot_components_vs_time(
        frames, gt, meas, pred if with_predicted else None,
        path=out, error_path=err_out, path_panel_path=path_out,
        title=("Итоговый результат: компоненты положения — "
               "эталон (—) и оценка по детекциям (---)"),
        fps=fps)
    plt.close("all")
    return True


def plot_coverage(evaluation: dict, out: Path) -> bool:
    counts = evaluation.get("coverage_counts")
    if not counts:
        return False
    labels = ["измерено", "только прогноз", "без оценки"]
    values = [counts["frames_measured"], counts["frames_predicted_only"],
              counts["frames_unobservable"]]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(labels, values, color=["#1b7f3b", "#d17b00", "#a03030"])
    for i, v in enumerate(values):
        ax.text(i, v, str(v), ha="center", va="bottom")
    ax.set_ylabel("кадров")
    ax.set_title(f"Покрытие кадров (всего {counts['frames_total']})")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Сводка и графики по truck-workflow")
    ap.add_argument("--detect-stats", default="out/truck/detect_stats.json")
    ap.add_argument("--anchor", default="datasets/truck/anchor_offset.json")
    ap.add_argument("--anchor-views", default="datasets/truck/anchor_offset_views.json")
    ap.add_argument("--visibility", default="out/experiments/visibility.json")
    ap.add_argument("--robustness", default="out/experiments/robustness.json")
    ap.add_argument("--detector-benchmark", default="out/experiments/detector_benchmark.json")
    ap.add_argument("--eval", default="out/truck/evaluation.json")
    ap.add_argument("--per-frame", default="out/truck/eval_per_frame.csv")
    ap.add_argument("--estimates", default="out/truck/estimates.csv")
    ap.add_argument("--root", default="huniashahatona/zadania")
    ap.add_argument("--repairs", default="datasets/truck/ground_truth_repairs.json")
    ap.add_argument("--out", default="out/truck/report")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    det_stats = read_json(Path(args.detect_stats))
    anchor = read_json(Path(args.anchor))
    anchor_views = read_json(Path(args.anchor_views))
    visibility = read_json(Path(args.visibility))
    robustness = read_json(Path(args.robustness))
    bench = read_json(Path(args.detector_benchmark))
    evaluation = read_json(Path(args.eval))
    rows = read_rows(Path(args.per_frame))

    graphs = []
    if plot_error_vs_frame(rows, out_dir / "error_vs_frame.png"):
        graphs.append("error_vs_frame.png")
    if plot_trajectory(rows, out_dir / "trajectory_xy.png"):
        graphs.append("trajectory_xy.png")
    fps = float((evaluation or {}).get("fps") or 30.0)
    gt_by_frame = None
    try:
        from multiview3d.truck_dataset import TruckDataset
        gt_by_frame = TruckDataset.discover(
            args.root, repairs_path=args.repairs or None).ground_truth_by_frame()
    except Exception as exc:  # нет данных — график построится по GT из CSV
        print(f"  GT датасета недоступен ({exc}); компоненты — по GT из CSV")
    if plot_components(rows, gt_by_frame, out_dir / "components_vs_time.png",
                       out_dir / "components_error_vs_time.png", fps=fps,
                       path_out=out_dir / "components_with_path.png"):
        graphs.append("components_vs_time.png")
        graphs.append("components_error_vs_time.png")
        graphs.append("components_with_path.png")
    if plot_components(rows, gt_by_frame,
                       out_dir / "components_vs_time_6lines.png",
                       out_dir / "components_error_vs_time_6lines.png",
                       fps=fps, with_predicted=False):
        graphs.append("components_vs_time_6lines.png")
        graphs.append("components_error_vs_time_6lines.png")
    if evaluation and plot_coverage(evaluation, out_dir / "coverage.png"):
        graphs.append("coverage.png")

    summary = {
        "artifacts": {
            "detect_stats": args.detect_stats if det_stats else None,
            "anchor_offset": args.anchor if anchor else None,
            "anchor_offset_views": args.anchor_views if anchor_views else None,
            "visibility": args.visibility if visibility else None,
            "robustness": args.robustness if robustness else None,
            "detector_benchmark": args.detector_benchmark if bench else None,
            "evaluation": args.eval if evaluation else None,
        },
        "detection": (None if not det_stats else {
            "frames_processed": det_stats["frames_processed"],
            "frames_with_detection": det_stats["frames_with_detection"],
            "frame_detection_rate": det_stats["frame_detection_rate"],
            "detections_total": det_stats["detections_total"],
            "per_camera_frame_rate": det_stats["per_camera_frame_rate"],
            "ms_per_frame": det_stats["ms_per_frame"],
            "fps_processing": det_stats["fps_processing"],
        }),
        "anchor": (None if not anchor else {
            "offset_m": anchor.get("offset_m"),
            "anchor": anchor.get("anchor"),
            "train_px_mean": (anchor.get("train") or {}).get("px_mean"),
            "holdout_px_mean": (anchor.get("holdout") or {}).get("px_mean"),
            "per_camera_holdout": anchor.get("per_camera_holdout"),
        }),
        "anchor_views": (None if not anchor_views else {
            "offsets_m": anchor_views.get("offsets_m"),
            "min_observations": anchor_views.get("min_observations"),
            "per_views": {
                k: {key: v.get(key) for key in
                    ("n", "mae_before_m", "mae_in_sample_m",
                     "cv_parity_mae_m", "cv_time_mae_m")}
                for k, v in (anchor_views.get("per_views") or {}).items()},
        }),
        "visibility": (None if not visibility else {
            "frames_analyzed": visibility["frames_analyzed"],
            "share_ge2_cameras": visibility["share_ge2_cameras"],
            "per_camera_share": visibility["per_camera_share"],
            "cameras_per_frame_histogram": visibility["cameras_per_frame_histogram"],
        }),
        "evaluation": (None if not evaluation else {
            "frame_range": evaluation.get("frame_range"),
            "measured": evaluation.get("measured"),
            "predicted": evaluation.get("predicted"),
            "coverage": evaluation.get("coverage"),
            "coverage_counts": evaluation.get("coverage_counts"),
            "per_camera_subset": evaluation.get("per_camera_subset"),
            "gt_invalid_frames": evaluation.get("gt_invalid_frames"),
        }),
        "robustness": (None if not robustness else {
            "noise": robustness.get("noise"),
            "calibration": robustness.get("calibration"),
            "timing": robustness.get("timing"),
        }),
        "detector_benchmark": (None if not bench else [
            {"model": r["model"], "imgsz": r["imgsz"],
             "detection_rate_overall": r["detection_rate_overall"],
             "ms_per_camera_frame": r["ms_per_camera_frame"]}
            for r in bench.get("runs", [])
        ]),
        "graphs": graphs,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Сводка truck-workflow")
    if summary["detection"]:
        d = summary["detection"]
        print(f"  детекция: кадров {d['frames_processed']}, с детекцией "
              f"{d['frames_with_detection']} ({100.0 * d['frame_detection_rate']:.1f} %), "
              f"{d['ms_per_frame']} мс/кадр")
    if summary["anchor"]:
        a = summary["anchor"]
        print(f"  якорь: offset={[round(v, 3) for v in a['offset_m']]}, "
              f"невязка train {a['train_px_mean']} px -> holdout {a['holdout_px_mean']} px")
    if summary["anchor_views"]:
        av = summary["anchor_views"]
        print(f"  якорь по наборам видов: {len(av['offsets_m'] or {})} наборов, "
              f"наборов в отчёте {len(av['per_views'] or {})}")
        for k, v in sorted((av["per_views"] or {}).items(),
                           key=lambda kv: -(kv[1]["n"] or 0))[:4]:
            print(f"    {k}: n={v['n']}, MAE {v['mae_before_m']} -> "
                  f"{v['mae_in_sample_m']} м (CV parity {v['cv_parity_mae_m']}, "
                  f"CV time {v['cv_time_mae_m']})")
    if summary["evaluation"]:
        e = summary["evaluation"]
        m = e["measured"]
        print(f"  оценка: MEASURED n={m['n_points']}, MAE={m['mae_mm']} мм, "
              f"RMSE={m['rmse_mm']}, P95={m['p95_mm']}, max={m['max_mm']}")
        print(f"  покрытие: {json.dumps(e['coverage'], ensure_ascii=False)}")
    if graphs:
        print(f"  графики: {', '.join(graphs)}")
    print(f"JSON: {out_dir / 'summary.json'}")
    missing = [k for k, v in summary["artifacts"].items() if v is None]
    if missing:
        print(f"  нет артефактов: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
