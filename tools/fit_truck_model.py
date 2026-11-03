"""Подгонка 3D-модели грузовика (FBX) к детекциям: проверка, объясняет ли модель боксы.

Зачем: детектор даёт центр bbox, GT — pivot актора, и расхождение 137 px даёт
метры ошибки. Если 3D-модель правильно ставится в мир (позиция GT + курс по
скорости), её проекция должна совпасть с наблюдаемыми bbox. Тогда:

* подтверждается геометрия/масштаб/ориентация модели;
* появляется «оракульный» детектор (боксы из модели) — верхняя граница
  точности конвейера;
* и псевдо-разметка для дообучения детектора на этой сцене.

Модель: Y-up (по габаритам: длина по Z, высота по Y, ширина по X), единицы —
сантиметры (6.68 × 1.68 × 2.42 м). Скрипт перебирает варианты осей и
подбирает: сдвиг модели относительно pivot, доворот курса и масштаб.

Запуск:
    python tools/fit_truck_model.py --detections out/truck/detections.csv `
        --out datasets/truck/model_alignment.json --max-obs 250
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from scipy.optimize import minimize

from multiview3d.camera import project_point
from multiview3d.truck_dataset import DEFAULT_REPAIRS, DEFAULT_ROOT, TruckDataset
from multiview3d.truck_pipeline import load_detections


def fbx_size(path: str | Path) -> np.ndarray:
    """Габариты модели (X, Y, Z) в метрах: FBX в сантиметрах.

    Обязательно ``Process_PreTransformVertices``: без него вершины остаются в
    локальных системах узлов и габарит получается неверным (проверено: 0.15 м
    вместо 1.68 м).
    """
    import assimp_py
    scene = assimp_py.import_file(
        str(path), assimp_py.Process_Triangulate | assimp_py.Process_PreTransformVertices)
    pts = [np.asarray(m.vertices, dtype=float).reshape(-1, 3) for m in scene.meshes]
    p = np.vstack([q for q in pts if q.size])
    size_cm = p.max(axis=0) - p.min(axis=0)
    return size_cm / 100.0


def heading_from_gt(gt: dict[int, np.ndarray]) -> dict[int, float]:
    """Курс по конечным разностям скорости (GT не содержит ориентации)."""
    frames = sorted(gt)
    pos = {f: np.asarray(gt[f], dtype=float) for f in frames}
    yaw = {}
    for i, f in enumerate(frames):
        a = pos[frames[max(i - 1, 0)]]
        b = pos[frames[min(i + 1, len(frames) - 1)]]
        d = b[:2] - a[:2]
        if np.linalg.norm(d) < 1e-6:
            yaw[f] = yaw.get(frames[i - 1], 0.0)
        else:
            yaw[f] = float(np.arctan2(d[1], d[0]))
    return yaw


def corners_world(size: np.ndarray, center: np.ndarray, yaw: float,
                  up_axis: str, fwd_axis: str, fwd_sign: float) -> np.ndarray:
    """8 углов 3D-бокса модели в мировых координатах (мир Unreal: Z вверх)."""
    sx, sy, sz = size
    half = {"X": sx / 2, "Y": sy / 2, "Z": sz / 2}
    axes = {"X": "XYZ"[0], "Y": "XYZ"[1], "Z": "XYZ"[2]}
    # локальные оси: up (мировая Z), forward (горизонт по yaw), lateral
    up = half[up_axis] * 2
    fwd = half[fwd_axis] * 2 * fwd_sign
    lat_axis = [a for a in "XYZ" if a not in (up_axis, fwd_axis)][0]
    lat = half[lat_axis] * 2
    cf, sf = np.cos(yaw), np.sin(yaw)
    pts = []
    for du, df, dh in itertools.product((-0.5, 0.5), repeat=3):
        lateral = du * lat
        forward = df * fwd
        height = dh * up
        wx = center[0] + lateral * (-sf) + forward * cf
        wy = center[1] + lateral * cf + forward * sf
        wz = center[2] + height
        pts.append((wx, wy, wz))
    return np.asarray(pts, dtype=float)


def project_box(cam, pts: np.ndarray) -> tuple[float, float, float, float] | None:
    uv = []
    for p in pts:
        try:
            u, v = project_point(cam, p)
        except ValueError:
            return None
        if not (np.isfinite(u) and np.isfinite(v)):
            return None
        uv.append((u, v))
    u = [p[0] for p in uv]
    v = [p[1] for p in uv]
    return min(u), min(v), max(u), max(v)


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Подгонка 3D-модели к детекциям")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--repairs", default=str(DEFAULT_REPAIRS))
    ap.add_argument("--model", default="models/CargoTruck.fbx")
    ap.add_argument("--detections", required=True)
    ap.add_argument("--train-start", type=int, default=0)
    ap.add_argument("--train-end", type=int, default=500)
    ap.add_argument("--max-obs", type=int, default=250)
    ap.add_argument("--up", default=None, choices=["X", "Y", "Z"],
                    help="ось модели, смотрящая вверх (по умолчанию — перебор)")
    ap.add_argument("--forward", default=None, choices=["X", "Y", "Z"],
                    help="ось модели по направлению движения")
    ap.add_argument("--forward-sign", type=float, default=1.0)
    ap.add_argument("--out", default="datasets/truck/model_alignment.json")
    args = ap.parse_args(argv)

    size = fbx_size(args.model)
    print(f"модель: {args.model}, габариты (м): X={size[0]:.3f} Y={size[1]:.3f} Z={size[2]:.3f}")

    ds = TruckDataset.discover(args.root, repairs_path=args.repairs or None)
    gt = ds.ground_truth_by_frame()
    yaw_gt = heading_from_gt(gt)
    dets = load_detections(args.detections)

    obs = []
    for f in sorted(dets):
        if not (args.train_start <= f <= args.train_end) or f not in gt:
            continue
        cams = ds.provider.calibrations_at(f, nearest=True)
        for name, items in dets[f].items():
            items = [d for d in items if d.bbox is not None]
            if not items or name not in cams:
                continue
            best = max(items, key=lambda d: d.score)
            obs.append({"frame": f, "camera": name, "cam": cams[name],
                        "box": tuple(float(v) for v in best.bbox),
                        "center": tuple(float(v) for v in best.center)})
    if len(obs) > args.max_obs:
        idx = np.linspace(0, len(obs) - 1, args.max_obs).astype(int)
        obs = [obs[i] for i in idx]
    print(f"наблюдений для фита: {len(obs)}")

    def mean_iou(up_axis, fwd_axis, fwd_sign, params) -> float:
        dx, dy, dz, dyaw, scale = params
        vals = []
        for o in obs:
            f = o["frame"]
            c = np.asarray(gt[f], dtype=float) + np.array([dx, dy, dz])
            pts = corners_world(size * scale, c, yaw_gt[f] + np.deg2rad(dyaw),
                                up_axis, fwd_axis, fwd_sign)
            box = project_box(o["cam"], pts)
            if box is None:
                continue
            vals.append(iou(box, o["box"]))
        return float(np.mean(vals)) if vals else 0.0

    # 1) грубый перебор осей и курса (или фиксированная физическая ориентация)
    best = None
    if args.up and args.forward:
        up_axis, fwd_axis, fwd_sign = args.up, args.forward, float(args.forward_sign)
        best = (0.0, up_axis, fwd_axis, fwd_sign, 0)
        print(f"ориентация задана вручную: up={up_axis}, forward={fwd_axis}"
              f"{'+' if fwd_sign > 0 else '-'}")
    for up_axis, fwd_axis in ([] if best else itertools.permutations("XYZ", 2)):
        for fwd_sign in (1.0, -1.0):
            for dyaw in (0, 90, -90, 180):
                score = mean_iou(up_axis, fwd_axis, fwd_sign, (0, 0, 0, dyaw, 1.0))
                if best is None or score > best[0]:
                    best = (score, up_axis, fwd_axis, fwd_sign, dyaw)
    score, up_axis, fwd_axis, fwd_sign, dyaw0 = best
    if not (args.up and args.forward):
        print(f"лучший грубый вариант: up={up_axis}, forward={fwd_axis}"
              f"{'+' if fwd_sign > 0 else '-'}, dyaw={dyaw0}°, IoU={score:.3f}")

    # 2) непрерывная подгонка с физическими границами (масштаб мира датасета
    # может отличаться от «метров» модели: сцена Unreal самосогласована, но её
    # масштаб произволен — поэтому scale ищем в широком диапазоне)
    x0 = np.array([0.0, 0.0, 0.0, float(dyaw0), 2.0])
    bounds = [(-8.0, 8.0), (-8.0, 8.0), (-4.0, 8.0), (-180.0, 180.0), (1.0, 6.0)]
    res = minimize(lambda p: -mean_iou(up_axis, fwd_axis, fwd_sign, p),
                   x0=x0, method="Powell", bounds=bounds,
                   options={"maxiter": 3000, "xtol": 1e-3, "ftol": 1e-5})
    p = res.x
    final = -float(res.fun)
    print(f"после фита: IoU={final:.3f}; сдвиг=({p[0]:+.2f}, {p[1]:+.2f}, {p[2]:+.2f}) м, "
          f"dyaw={p[3]:+.1f}°, scale={p[4]:.3f}")

    per_cam = {}
    for cam in ds.names:
        sub = [o for o in obs if o["camera"] == cam]
        if not sub:
            continue
        obs_cam = obs
        obs = sub
        per_cam[cam] = round(mean_iou(up_axis, fwd_axis, fwd_sign, p), 3)
        obs = obs_cam
    print("по камерам IoU:", per_cam)

    doc = {
        "model": str(args.model).replace("\\", "/"),
        "model_size_m": {"X": round(float(size[0]), 4), "Y": round(float(size[1]), 4),
                         "Z": round(float(size[2]), 4)},
        "axes": {"up": up_axis, "forward": fwd_axis, "forward_sign": fwd_sign},
        "offset_m": [round(float(v), 4) for v in p[:3]],
        "dyaw_deg": round(float(p[3]), 3),
        "scale": round(float(p[4]), 5),
        "iou_mean": round(final, 4),
        "iou_per_camera": per_cam,
        "observations": len(obs),
        "train_frames": [args.train_start, args.train_end],
        "detections_csv": str(args.detections).replace("\\", "/"),
        "note": ("Сдвиг — положение начала координат модели относительно GT-pivot; "
                 "курс взят из разности GT-позиций (ориентации в GT нет). "
                 "IoU считается по проекции 8 углов 3D-бокса модели."),
    }
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"записано: {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
