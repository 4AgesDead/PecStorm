"""Fit 3D-модели CargoTruck под scene2 по детекциям и текущей траектории.

GT в scene2 нет, поэтому «псевдо-GT» — submission_scene2.csv (PCHIP).
Подбираем offset/scale/dyaw/pivot_frac, максимизируя IoU проекции модели
с боксами COCO. Это калибровка модели, а не измерение.
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


def yaw_at(rows: dict[int, tuple[float, float]], f: int) -> float:
    a = np.array(rows[max(0, f - 5)], dtype=float)
    b = np.array(rows[min(1931, f + 5)], dtype=float)
    d = b - a
    return float(np.arctan2(d[1], d[0])) if np.linalg.norm(d) > 1e-6 else 0.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fit модели под scene2")
    ap.add_argument("--submission", default="results/scene2/submission_scene2.csv")
    ap.add_argument("--detections", default="out/scene2/dets_fit.csv")
    ap.add_argument("--base-alignment", default="datasets/truck/model_alignment.json")
    ap.add_argument("--out", default="datasets/scene2/model_alignment_scene2.json")
    ap.add_argument("--max-obs", type=int, default=220)
    ap.add_argument("--near-px", type=float, default=420.0)
    args = ap.parse_args(argv)

    ds = Scene2Dataset.discover("scene2")
    base = json.loads(Path(args.base_alignment).read_text(encoding="utf-8"))
    size0 = np.array([base["model_size_m"][k] for k in "XYZ"], dtype=float)
    up, fwd = base["axes"]["up"], base["axes"]["forward"]
    sign = float(base["axes"]["forward_sign"])

    pos: dict[int, tuple[float, float, float]] = {}
    with Path(args.submission).open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            pos[int(r["frame"])] = (float(r["x"]), float(r["y"]), float(r["z"]))
    xy = {f: (p[0], p[1]) for f, p in pos.items()}

    best: dict[tuple[int, str], dict] = {}
    with Path(args.detections).open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            key = (int(r["frame"]), r["camera"])
            cur = best.get(key)
            if cur is None or float(r["score"]) > float(cur["score"]):
                best[key] = r

    obs = []
    for (f, cam_name), r in best.items():
        if f not in pos or cam_name not in ds.names:
            continue
        cam = ds.provider.calibration(cam_name, f)
        p = np.array(pos[f], dtype=float)
        try:
            u, v = project_point(cam, p)
        except ValueError:
            continue
        box = (float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"]))
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        if np.hypot(cx - u, cy - v) > args.near_px:
            continue
        obs.append({"frame": f, "camera": cam_name, "cam": cam, "box": box})
    if len(obs) > args.max_obs:
        idx = np.linspace(0, len(obs) - 1, args.max_obs).astype(int)
        obs = [obs[i] for i in idx]
    print(f"наблюдений для fit: {len(obs)}")
    if len(obs) < 10:
        print("мало наблюдений", file=sys.stderr)
        return 2

    def score(params) -> float:
        dx, dy, dz, dyaw, scale, beta = params
        vals = []
        for o in obs:
            p = np.array(pos[o["frame"]], dtype=float) + np.array([dx, dy, dz])
            pts = corners_world(size0 * scale, p,
                                yaw_at(xy, o["frame"]) + np.deg2rad(dyaw),
                                up, fwd, sign, pivot_frac=beta)
            box = project_box(o["cam"], pts)
            if box is not None:
                vals.append(iou(box, o["box"]))
        return float(np.mean(vals)) if vals else 0.0

    init = np.array([
        float(base["offset_m"][0]), float(base["offset_m"][1]),
        float(base["offset_m"][2]), float(base["dyaw_deg"]),
        float(base["scale"]), float(base.get("pivot_frac", 0.4)),
    ])
    print(f"старт IoU={score(init):.3f}, params={np.round(init, 3)}")
    res = minimize(lambda p: -score(p), init, method="Powell",
                   bounds=[(-15, 15), (-15, 15), (-8, 8), (-180, 180),
                           (1.0, 8.0), (0.0, 0.6)],
                   options={"maxiter": 3000, "xtol": 1e-3, "ftol": 1e-5})
    p = res.x
    final = -float(res.fun)
    print(f"fit IoU={final:.3f}, params={np.round(p, 3)}")

    per_cam = {}
    for cam_name in ds.names:
        sub = [o for o in obs if o["camera"] == cam_name]
        if not sub:
            continue
        old = obs
        obs = sub
        per_cam[cam_name] = round(score(p), 3)
        obs = old
    print("IoU по камерам:", per_cam)

    # 2D-сдвиг «центр бокса -> пивот» по камерам (по псевдо-GT)
    shifts: dict[str, list[float]] = {}
    for o in obs:
        try:
            pu, pv = project_point(o["cam"], np.array(pos[o["frame"]], dtype=float))
        except ValueError:
            continue
        box = o["box"]
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        shifts.setdefault(o["camera"], []).append([pu - cx, pv - cy])
    px_offsets = {}
    for name, vals in shifts.items():
        if len(vals) >= 3:
            a = np.median(np.asarray(vals), axis=0)
            px_offsets[name] = [round(float(a[0]), 2), round(float(a[1]), 2)]
    print("2D-сдвиги (px):", px_offsets)

    doc = {
        "model": base["model"],
        "model_size_m": base["model_size_m"],
        "axes": base["axes"],
        "offset_m": [round(float(v), 4) for v in p[:3]],
        "dyaw_deg": round(float(p[3]), 3),
        "scale": round(float(p[4]), 5),
        "pivot_frac": round(float(p[5]), 4),
        "iou_mean": round(final, 4),
        "iou_per_camera": per_cam,
        "px_offsets_px": px_offsets,
        "observations": len(obs),
        "pseudo_gt": args.submission,
        "detections": args.detections,
        "note": "Калибровка модели по псевдо-GT scene2 (PCHIP) и COCO-боксам; GT нет.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"сохранено: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
