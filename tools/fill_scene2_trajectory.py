"""Заполнить все 1932 кадра траекторией scene2 (без GT).

Берём только MEASURED-точки из `out/scene2/est_scene2_step2.csv`, сортируем по
кадру и интерполируем PCHIP; вне диапазона — линейная экстраполяция. Затем
лёгкое сглаживание Савицкого–Голея, чтобы убрать дрожание между сегментами.

Это НЕ измерение на каждом кадре: статус `INTERPOLATED` означает, что 3D там
восстановлен по соседним измерениям, а не получен триангуляцией.
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

import numpy as np
from scipy.interpolate import PchipInterpolator, UnivariateSpline
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter


def read_measured(path: Path, min_track_points: int = 1) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("status") != "MEASURED":
                continue
            try:
                x, y, z = float(r["x"]), float(r["y"]), float(r["z"])
            except (KeyError, TypeError, ValueError):
                continue
            if not all(np.isfinite(v) for v in (x, y, z)):
                continue
            rows.append({"frame": int(r["frame"]), "x": x, "y": y, "z": z,
                         "track_id": r.get("track_id", ""),
                         "views": r.get("views", ""),
                         "confidence": r.get("confidence", ""),
                         "mean_reproj_px": r.get("mean_reproj_px", "")})
    rows.sort(key=lambda r: r["frame"])
    if min_track_points > 1:
        counts = {}
        for r in rows:
            counts[r["track_id"]] = counts.get(r["track_id"], 0) + 1
        rows = [r for r in rows if counts.get(r["track_id"], 0) >= min_track_points]
    # одна точка на кадр
    out = []
    for r in rows:
        if out and r["frame"] == out[-1]["frame"]:
            out[-1] = r
        else:
            out.append(r)
    return out


def local_consistency(frames: np.ndarray, arr: np.ndarray, window: int = 30,
                      min_neighbors: int = 4) -> np.ndarray:
    """Невязка точки относительно медианы соседей по времени (окно ±window).

    Глобальный сплайн для отбраковки не годится: на резких поворотах он
    отстаёт и «выбросом» выглядит честное измерение. Локальная медиана
    повторяет поворот и ловит именно одиночные срывы трекера.
    """
    n = len(frames)
    res = np.full(n, np.inf, dtype=float)
    for i in range(n):
        lo = int(np.searchsorted(frames, frames[i] - window, "left"))
        hi = int(np.searchsorted(frames, frames[i] + window, "right"))
        idx = [j for j in range(lo, hi) if j != i]
        if len(idx) < min_neighbors:
            continue
        med = np.median(arr[idx], axis=0)
        res[i] = float(np.linalg.norm(arr[i] - med))
    return res


def isolated_mask(frames: np.ndarray, arr: np.ndarray, frame_gap: int = 10,
                  max_dist_m: float = 40.0) -> np.ndarray:
    """Точки, у которых нет соседей дальше ±frame_gap кадров ближе max_dist_m.

    Так отсекается сегмент, который целиком «улетел» на другой объект: он
    связен сам с собой, но оторван от остальной траектории.
    """
    n = len(frames)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        far = np.abs(frames - frames[i]) > frame_gap
        if not far.any():
            continue
        d = np.linalg.norm(arr[far] - arr[i], axis=1).min()
        mask[i] = d <= max_dist_m
    return mask


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Заполнить траекторию scene2")
    ap.add_argument("--estimates", default="out/scene2/est_scene2_step2.csv")
    ap.add_argument("--out-csv", default="out/scene2/submission_scene2.csv")
    ap.add_argument("--out-json", default="out/scene2/submission_scene2.json")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=1931)
    ap.add_argument("--smooth-window", type=int, default=0,
                    help="окно Савицкого–Голея (0 — не применять)")
    ap.add_argument("--smooth-sigma", type=float, default=16.0,
                    help="сигма гауссова сглаживания по кадрам (0 — выключить)")
    ap.add_argument("--avg-half-frames", type=int, default=2,
                    help="усреднить измерения в окне ±N кадров (0 — не усреднять)")
    ap.add_argument("--min-track-points", type=int, default=0,
                    help="отбросить короткие треки (0 — не фильтровать)")
    ap.add_argument("--trim-quantile", type=float, default=0.0,
                    help="срезать крайние q по X/Y (0 — выключить)")
    ap.add_argument("--robust", dest="robust", action="store_true", default=True,
                    help="отбросить точки, не согласующиеся с соседями по времени")
    ap.add_argument("--no-robust", dest="robust", action="store_false")
    ap.add_argument("--isolated-max-dist", type=float, default=0.0,
                    help="отсечь точки без соседей ближе N м (0 — выключить); "
                         "по умолчанию выключено: проверено, что ранний кластер "
                         "кадров 41..46 реален (аэрокамера cam2 видит там грузовик)")
    ap.add_argument("--max-residual-m", type=float, default=6.0,
                    help="порог локальной невязки, метры")
    ap.add_argument("--window-frames", type=int, default=30,
                    help="полуокно локальной проверки, кадры")
    args = ap.parse_args(argv)

    pts = read_measured(Path(args.estimates), args.min_track_points)
    if args.trim_quantile and len(pts) >= 20:
        xs = np.array([p["x"] for p in pts])
        ys = np.array([p["y"] for p in pts])
        x0, x1 = np.quantile(xs, [args.trim_quantile, 1 - args.trim_quantile])
        y0, y1 = np.quantile(ys, [args.trim_quantile, 1 - args.trim_quantile])
        before = len(pts)
        pts = [p for p in pts if x0 <= p["x"] <= x1 and y0 <= p["y"] <= y1]
        print(f"trim {args.trim_quantile:.3f}: {before} -> {len(pts)} точек")
    if len(pts) < 3:
        print("мало MEASURED-точек", file=sys.stderr)
        return 2
    if args.robust and len(pts) >= 20:
        f0 = np.array([p["frame"] for p in pts], dtype=float)
        a0 = np.array([[p["x"], p["y"], p["z"]] for p in pts], dtype=float)
        keep = np.ones(len(pts), dtype=bool)
        if args.isolated_max_dist > 0:
            keep = isolated_mask(f0, a0, max_dist_m=args.isolated_max_dist)
            print(f"isolated: отброшено {int((~keep).sum())} точек "
                  f"(нет соседей ближе {args.isolated_max_dist:.0f} м)")
        res = local_consistency(f0, a0, window=args.window_frames)
        finite = np.isfinite(res)
        if finite.any():
            print(f"локальная невязка: медиана {np.median(res[finite]):.2f} м, "
                  f"P90 {np.percentile(res[finite], 90):.2f} м")
        keep &= ~(np.isfinite(res) & (res > args.max_residual_m))
        dropped = int((~keep).sum())
        if dropped:
            print(f"robust: отброшено {dropped} точек из {len(pts)}")
            pts = [p for p, k in zip(pts, keep) if k]
    frames = np.array([p["frame"] for p in pts], dtype=float)
    arr = np.array([[p["x"], p["y"], p["z"]] for p in pts], dtype=float)
    if args.avg_half_frames > 0 and len(pts) >= 3:
        raw = arr.copy()
        arr = raw.copy()
        for i in range(len(frames)):
            sel = np.abs(frames - frames[i]) <= args.avg_half_frames
            if int(sel.sum()) >= 2:
                arr[i] = raw[sel].mean(axis=0)
        print(f"усреднение ±{args.avg_half_frames} кадра: "
              f"{int((np.abs(arr - raw).max(axis=1) > 0).sum())} точек сглажено")
    print(f"MEASURED: {len(pts)} точек, кадры {int(frames[0])}..{int(frames[-1])}")

    all_frames = np.arange(args.start, args.end + 1, dtype=float)
    interp = np.empty((len(all_frames), 3), dtype=float)
    for d in range(3):
        f = PchipInterpolator(frames, arr[:, d], extrapolate=False)
        vals = f(all_frames)
        # края: линейная экстраполяция по двум крайним точкам
        left = all_frames < frames[0]
        right = all_frames > frames[-1]
        if left.any():
            vals[left] = arr[0, d]
        if right.any():
            vals[right] = arr[-1, d]
        interp[:, d] = vals

    # грузовик на земле: Z не должен улетать в 10+ м из-за экстраполяции
    interp[:, 2] = np.clip(interp[:, 2], 0.5, 8.0)

    if args.smooth_window and args.smooth_window % 2 == 1:
        win = min(args.smooth_window, len(all_frames) // 2 * 2 - 1)
        if win >= 5:
            interp = savgol_filter(interp, win, 2, axis=0)

    if args.smooth_sigma and args.smooth_sigma > 0:
        # гауссово сглаживание вместо SG: не даёт «звона» на изломах
        interp = gaussian_filter1d(interp, args.smooth_sigma, axis=0, mode="nearest")
        interp[:, 2] = np.clip(interp[:, 2], 0.5, 8.0)

    measured_frames = set(int(f) for f in frames)
    measured_by_frame = {p["frame"]: p for p in pts}
    vel = np.gradient(interp, axis=0) * 30.0
    rows = []
    for i, (f, (x, y, z)) in enumerate(zip(all_frames.astype(int), interp)):
        p = measured_by_frame.get(int(f))
        views = p["views"] if p else ""
        rows.append({
            "frame": int(f),
            "timestamp": round(f / 30.0, 6),
            "x": round(float(x), 4),
            "y": round(float(y), 4),
            "z": round(float(z), 4),
            "vx": round(float(vel[i, 0]), 4),
            "vy": round(float(vel[i, 1]), 4),
            "vz": round(float(vel[i, 2]), 4),
            "status": "MEASURED" if f in measured_frames else "INTERPOLATED",
            "method": "triangulation" if f in measured_frames else "pchip",
            "n_views": len([v for v in views.split("|") if v]) if views else 0,
            "views": views,
            "mean_reproj_px": p["mean_reproj_px"] if p else "",
            "confidence": p["confidence"] if p else "",
        })

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # минимальный вариант: только frame,x,y,z — на случай строгого парсера
    min_out = out.with_name(out.stem + "_xyz.csv")
    try:
        with min_out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["frame", "x", "y", "z"])
            writer.writeheader()
            writer.writerows([{k: r[k] for k in ("frame", "x", "y", "z")} for r in rows])
    except PermissionError:
        alt = min_out.with_name(min_out.stem + "_new.csv")
        with alt.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["frame", "x", "y", "z"])
            writer.writeheader()
            writer.writerows([{k: r[k] for k in ("frame", "x", "y", "z")} for r in rows])
        print(f"ВНИМАНИЕ: {min_out} занят (Excel?); записан {alt}", file=sys.stderr)

    d_xy = np.linalg.norm(np.diff(interp[:, :2], axis=0), axis=1)
    meta = {
        "source": args.estimates,
        "measured_points": len(pts),
        "measured_range": [int(frames[0]), int(frames[-1])],
        "frames": len(rows),
        "frame_range": [args.start, args.end],
        "path_length_xy_m": round(float(d_xy.sum()), 2),
        "method": ("PCHIP по MEASURED (устойчивый отсев выбросов) "
                   "+ константные края + SG-сглаживание"),
        "warning": "INTERPOLATED — не измерение; GT в scene2 нет",
    }
    Path(args.out_json).write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    print(f"кадров: {len(rows)}; путь XY: {meta['path_length_xy_m']} м")
    print(f"CSV: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
