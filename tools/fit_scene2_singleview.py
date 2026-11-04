"""Пилот single-view pose fit: bbox одной камеры + модель + плоскость Z=3.

Проверяем, можно ли по одному bbox получить 3D-позу грузовика. Если да —
это путь поднять покрытие scene2 без второго вида.
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
from scipy.optimize import minimize

from multiview3d.camera import project_point
from multiview3d.scene2_dataset import Scene2Dataset
from tools.fit_truck_model import corners_world, iou, project_box


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Single-view fit scene2")
    ap.add_argument("--submission", default="results/scene2/submission_scene2.csv")
    ap.add_argument("--detections", default="out/scene2/dets_fit.csv")
    ap.add_argument("--measured", default="out/scene2/est_scene2_step1.csv")
    ap.add_argument("--alignment", default="datasets/scene2/model_alignment_scene2.json")
    ap.add_argument("--near-px", type=float, default=420.0)
    ap.add_argument("--z", type=float, default=3.0)
    args = ap.parse_args(argv)

    ds = Scene2Dataset.discover("scene2")
    al = json.loads(Path(args.alignment).read_text(encoding="utf-8"))
    size = np.array([al["model_size_m"][k] for k in "XYZ"], dtype=float) * al["scale"]
    off = np.array(al["offset_m"], dtype=float)
    up, fwd = al["axes"]["up"], al["axes"]["forward"]
    sign = al["axes"]["forward_sign"]
    beta = al["pivot_frac"]
    dyaw = np.deg2rad(al["dyaw_deg"])

    pos = {}
    with Path(args.submission).open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            pos[int(r["frame"])] = np.array([float(r["x"]), float(r["y"]),
                                             float(r["z"])])

    best = {}
    with Path(args.detections).open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            key = (int(r["frame"]), r["camera"])
            cur = best.get(key)
            if cur is None or float(r["score"]) > float(cur["score"]):
                best[key] = r

    def yaw0(f):
        a = pos.get(max(0, f - 5), pos[f])
        b = pos.get(min(1931, f + 5), pos[f])
        d = b[:2] - a[:2]
        return float(np.arctan2(d[1], d[0])) if np.linalg.norm(d) > 1e-6 else 0.0

    def fit_one(f, cam_name, box):
        cam = ds.provider.calibration(cam_name, f)
        p0 = pos[f]
        try:
            u0, v0 = project_point(cam, p0)
        except ValueError:
            return None
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        if np.hypot(cx - u0, cy - v0) > args.near_px:
            return None
        y0 = yaw0(f)

        def obj(p):
            x, y, yaw = p
            center = np.array([x, y, args.z]) + off
            pts = corners_world(size, center, yaw + dyaw, up, fwd, sign,
                                pivot_frac=beta)
            q = project_box(cam, pts)
            return 1.0 - (iou(q, box) if q is not None else 0.0)

        res = minimize(obj, [p0[0], p0[1], y0], method="Powell",
                       bounds=[(p0[0] - 40, p0[0] + 40),
                               (p0[1] - 40, p0[1] + 40),
                               (y0 - np.pi, y0 + np.pi)],
                       options={"maxiter": 200, "xtol": 1e-2, "ftol": 1e-4})
        return float(1.0 - res.fun), res.x, p0

    rows = []
    for (f, cam), r in best.items():
        box = (float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"]))
        out = fit_one(f, cam, box)
        if out:
            score, p, p0 = out
            rows.append((f, cam, score, p[0], p[1], p[2], *p0))
    if not rows:
        print("нет fit-наблюдений", file=sys.stderr)
        return 2
    scores = np.array([r[2] for r in rows])
    shifts = np.array([np.hypot(r[3] - r[6], r[4] - r[7]) for r in rows])
    print(f"наблюдений: {len(rows)}")
    print(f"IoU: median {np.median(scores):.3f}, mean {scores.mean():.3f}, "
          f"P10 {np.percentile(scores, 10):.3f}, >0.5: {(scores > 0.5).mean():.2f}")
    print(f"сдвиг от псевдо-GT: median {np.median(shifts):.1f} м, "
          f"P90 {np.percentile(shifts, 90):.1f} м")
    measured = {}
    if Path(args.measured).exists():
        with Path(args.measured).open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r.get("status") == "MEASURED":
                    measured[int(r["frame"])] = np.array(
                        [float(r["x"]), float(r["y"]), float(r["z"])])
    common = [(r[3], r[4], measured[r[0]][0], measured[r[0]][1])
              for r in rows if r[0] in measured]
    if common:
        err = np.array([np.hypot(a - c, b - d) for a, b, c, d in common])
        print(f"single-view vs multi-view 3D: n={len(err)}, median {np.median(err):.1f} м, "
              f"P90 {np.percentile(err, 90):.1f} м")
    for r in sorted(rows, key=lambda x: -x[2])[:10]:
        print(" f", r[0], r[1], "IoU", round(r[2], 2),
              "fit", np.round(r[3:6], 1), "было", np.round(r[6:9], 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
