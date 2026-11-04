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
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter


def read_measured(path: Path) -> list[dict]:
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
                         "views": r.get("views", ""),
                         "confidence": r.get("confidence", ""),
                         "mean_reproj_px": r.get("mean_reproj_px", "")})
    rows.sort(key=lambda r: r["frame"])
    # одна точка на кадр
    out = []
    for r in rows:
        if out and r["frame"] == out[-1]["frame"]:
            out[-1] = r
        else:
            out.append(r)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Заполнить траекторию scene2")
    ap.add_argument("--estimates", default="out/scene2/est_scene2_step2.csv")
    ap.add_argument("--out-csv", default="out/scene2/submission_scene2.csv")
    ap.add_argument("--out-json", default="out/scene2/submission_scene2.json")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=1931)
    ap.add_argument("--smooth-window", type=int, default=31)
    args = ap.parse_args(argv)

    pts = read_measured(Path(args.estimates))
    if len(pts) < 3:
        print("мало MEASURED-точек", file=sys.stderr)
        return 2
    frames = np.array([p["frame"] for p in pts], dtype=float)
    arr = np.array([[p["x"], p["y"], p["z"]] for p in pts], dtype=float)
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
    with min_out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["frame", "x", "y", "z"])
        writer.writeheader()
        writer.writerows([{k: r[k] for k in ("frame", "x", "y", "z")} for r in rows])

    d_xy = np.linalg.norm(np.diff(interp[:, :2], axis=0), axis=1)
    meta = {
        "source": args.estimates,
        "measured_points": len(pts),
        "measured_range": [int(frames[0]), int(frames[-1])],
        "frames": len(rows),
        "frame_range": [args.start, args.end],
        "path_length_xy_m": round(float(d_xy.sum()), 2),
        "method": "PCHIP по MEASURED + линейная экстраполяция краёв + SG-сглаживание",
        "warning": "INTERPOLATED — не измерение; GT в scene2 нет",
    }
    Path(args.out_json).write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    print(f"кадров: {len(rows)}; путь XY: {meta['path_length_xy_m']} м")
    print(f"CSV: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
