"""Покадровая подгонка модели: проверка, переносится ли уточнение между видами.

Идея (шаг «лучше псевдо-разметка»): глобальные параметры модели (β, масштаб,
сдвиг) не описывают покадровые колебания позы. Если подогнать их по боксам
одной камеры, станет ли проекция точнее на **другой** камере того же кадра?

Это честная проверка: бокс второй камеры в фите не участвует. Если перенос
есть — уточнённые метки осмысленны и их стоит отдавать ученику; если нет —
уточнение подгоняется под шум детекций и смысла не имеет.
"""

import json
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from multiview3d.truck_dataset import DEFAULT_REPAIRS, DEFAULT_ROOT, TruckDataset
from multiview3d.truck_pipeline import load_detections
from tools.fit_truck_model import corners_world, fbx_size, heading_from_gt, iou, project_box

ds = TruckDataset.discover(DEFAULT_ROOT, repairs_path=DEFAULT_REPAIRS)
gt = ds.ground_truth_by_frame()
yaw = heading_from_gt(gt)
dets = load_detections("out/truck/detections_ens.csv")
al = json.load(open("datasets/truck/model_alignment.json", encoding="utf-8"))
size = fbx_size("models/CargoTruck.fbx") * al["scale"]
off = np.asarray(al["offset_m"], dtype=float)
beta0 = float(al.get("pivot_frac", 0.0))
dyaw = np.deg2rad(float(al["dyaw_deg"]))
up, fwd, sign = al["axes"]["up"], al["axes"]["forward"], al["axes"]["forward_sign"]
shift = {k: np.asarray(v, dtype=float) for k, v in (al.get("camera_shift_px") or {}).items()}


def project(cam, f, dbeta, dscale, dz):
    c = np.asarray(gt[f], dtype=float) + off + np.array([0.0, 0.0, dz])
    pts = corners_world(size * dscale, c, yaw[f] + dyaw, up, fwd, sign,
                        pivot_frac=beta0 + dbeta)
    b = project_box(cam, pts)
    return b


rows = []
for f in sorted(dets):
    if f not in gt:
        continue
    cams = ds.provider.calibrations_at(f, nearest=True)
    per = {}
    for name, items in dets[f].items():
        items = [d for d in items if d.bbox is not None]
        if items and name in cams:
            per[name] = (max(items, key=lambda d: d.score), cams[name])
    if len(per) >= 2:
        rows.append((f, per))
print(f"кадров с ≥2 видами и детекциями: {len(rows)}")


def box_with_shift(b, name):
    if b is None:
        return None
    s = shift.get(name)
    if s is None:
        return b
    return (b[0] + s[0], b[1] + s[1], b[2] + s[0], b[3] + s[1])


def fit_frame(f, per, fit_names):
    def loss(p):
        dbeta, dscale, dz = p
        vals = []
        for name in fit_names:
            det, cam = per[name]
            b = box_with_shift(project(cam, f, dbeta, dscale, dz), name)
            if b is None:
                continue
            vals.append(iou(b, det.bbox))
        if not vals:
            return 1.0
        return 1.0 - float(np.mean(vals)) + 0.05 * (dbeta ** 2 + (dscale - 1) ** 2 + 0.01 * dz ** 2)

    r = minimize(loss, np.zeros(3), method="Powell",
                 bounds=[(-0.2, 0.2), (0.9, 1.1), (-3.0, 3.0)],
                 options={"maxiter": 200, "xtol": 1e-3, "ftol": 1e-4})
    return r.x


before, after = [], []
for f, per in rows:
    names = sorted(per)
    fit_name, test_name = names[0], names[1]      # честное разделение видов
    p = fit_frame(f, per, [fit_name])
    for name, target in ((fit_name, "fit"), (test_name, "test")):
        det, cam = per[name]
        b0 = box_with_shift(project(cam, f, 0.0, 1.0, 0.0), name)
        b1 = box_with_shift(project(cam, f, *p), name)
        if b0 is None or b1 is None:
            continue
        v0, v1 = iou(b0, det.bbox), iou(b1, det.bbox)
        if target == "fit":
            before.append(v0)
            after.append(v1)
        else:
            before.append(v0)
            after.append(v1)

before = np.array(before)
after = np.array(after)
print(f"наблюдений (fit+test): {len(before)}")
print(f"IoU до уточнения:     mean={before.mean():.3f}, медиана={np.median(before):.3f}")
print(f"IoU после уточнения:  mean={after.mean():.3f}, медиана={np.median(after):.3f}")
print(f"доля кадров, где стало лучше: {(after > before + 0.01).mean():.3f}, "
      f"хуже: {(after < before - 0.01).mean():.3f}")
