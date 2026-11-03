"""Фит 3D-смещения якоря: pivot+delta должен проецироваться в точки касания колёс.

Если delta согласован (малые невязки во всех камерах) — конвенция координат
подтверждена, а якорь = GT pivot + delta (документируется как provisional anchor).
"""
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")
from multiview3d.unreal_camera import parse_camera_csv, parse_track_csv, CM_TO_M

DATA = Path("huniashahatona/zadania")

# точка касания колёс с дорогой (full-res), вручную по просмотренным кадрам
ANCHORS = {
    (1, 0):   (222, 505),
    (2, 300): (835, 540),
    (2, 450): (820, 530),
    (4, 900): (490, 660),
    (5, 750): (1310, 700),
    (3, 450): (1755, 580),
}

cams = {}
for i in range(1, 6):
    pr = parse_camera_csv(DATA / f"xyzpowcameta/object_transform_CineCameraActor{i}.csv")
    cams[i] = ({r.frame: r for r in pr.rows}, pr.focal_mm, pr.sensor_mm)
gt = parse_track_csv(DATA / "xyzpowcameta/object_transform_SM_AA_track.csv")
gt_by = {r.frame: np.array(r.loc) * CM_TO_M for r in gt.rows}


def axes(pitch, yaw, roll, sp, sy, so):
    s_p, c_p = np.sin(np.radians(sp * pitch)), np.cos(np.radians(pitch))
    s_y, c_y = np.sin(np.radians(sy * yaw)), np.cos(np.radians(yaw))
    s_o, c_o = np.sin(np.radians(so * roll)), np.cos(np.radians(roll))
    fwd = np.array([c_p * c_y, c_p * s_y, s_p])
    right = np.array([-s_o * s_p * c_y + c_o * s_y, -s_o * s_p * s_y - c_o * c_y, s_o * c_p])
    up = np.array([c_o * s_p * c_y + s_o * s_y, c_o * s_p * s_y - s_o * c_y, c_o * c_p])
    return fwd, right, up


def pixel(cam_i, frame, pt_m, sr, sp, sy, so):
    rows, focal, sensor = cams[cam_i]
    row = rows[frame]
    d = pt_m - np.array(row.loc) * CM_TO_M       # pivot->camera? camera->pivot
    fwd, right, up = axes(row.rot[1], row.rot[2], row.rot[0], sp, sy, so)
    pl = np.array([sr * (right @ d), -up @ d, fwd @ d])
    if pl[2] <= 0:
        return np.array([np.nan, np.nan])
    fx = focal * 1920 / sensor[0]; fy = focal * 1080 / sensor[1]
    return np.array([fx * pl[0] / pl[2] + 960, fy * pl[1] / pl[2] + 540])


def fit(sr, sp, sy, so):
    def resid(delta):
        out = []
        for (ci, f), (ua, va) in ANCHORS.items():
            p = pixel(ci, f, gt_by[f] + delta, sr, sp, sy, so)
            out.extend([p[0] - ua, p[1] - va])
        return np.array(out)
    res = least_squares(resid, x0=np.zeros(3), loss="soft_l1", f_scale=10.0)
    r = resid(res.x)
    return res.x, float(np.max(np.abs(r))), r


for so in (1.0, -1.0):
    delta, maxr, r = fit(-1.0, -1.0, 1.0, so)
    print(f"so={so:+.0f}: delta=({delta[0]:.2f}, {delta[1]:.2f}, {delta[2]:.2f}) м, "
          f"max|res|={maxr:.1f}px")
    print(f"   residuals per sample: {np.round(r.reshape(-1,2),1).tolist()}")
