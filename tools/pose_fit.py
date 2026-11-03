"""Fit per-camera pose (yaw/pitch offsets + global focal scale) from truck blobs.

Anchors = bottom-center of truck blob (full-res 1920x1080), from frame diff.
Convention under test: sr=-1 (right flipped), sp=-1 (pitch sin flipped), sy=+1, so=+1.
If all static cams agree on similar corrections, the convention is settled.
"""
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")
from multiview3d.unreal_camera import parse_camera_csv, parse_track_csv, CM_TO_M

DATA = Path("huniashahatona/zadania")

ANCHORS = {
    1: [(0, 330, 700), (120, 1306, 739)],
    2: [(120, 518, 1026), (300, 776, 674), (420, 792, 549), (450, 772, 526)],
    3: [(420, 1802, 605), (540, 1007, 579), (900, 492, 512)],
    4: [(300, 1810, 533), (750, 1400, 724), (900, 474, 1042)],
    5: [(750, 1310, 700)],
}

cams = {}
for i in range(1, 6):
    pr = parse_camera_csv(DATA / f"xyzpowcameta/object_transform_CineCameraActor{i}.csv")
    cams[i] = ({r.frame: r for r in pr.rows}, pr.focal_mm, pr.sensor_mm)
gt = parse_track_csv(DATA / "xyzpowcameta/object_transform_SM_AA_track.csv")
gt_by = {r.frame: np.array(r.loc) * CM_TO_M for r in gt.rows}


def axes(pitch, yaw, roll):
    s_p, c_p = np.sin(np.radians(pitch)), np.cos(np.radians(pitch))
    s_y, c_y = np.sin(np.radians(yaw)), np.cos(np.radians(yaw))
    s_o, c_o = np.sin(np.radians(roll)), np.cos(np.radians(roll))
    fwd = np.array([c_p * c_y, c_p * s_y, s_p])
    right = -np.array([-s_o * s_p * c_y + c_o * s_y, -s_o * s_p * s_y - c_o * c_y, s_o * c_p])
    up = np.array([c_o * s_p * c_y + s_o * s_y, c_o * s_p * s_y - s_o * c_y, c_o * c_p])
    return fwd, right, up


def pixel(cam_i, frame, pt, d_yaw, d_pitch, s):
    rows, focal, sensor = cams[cam_i]
    row = rows[frame]
    # effective angles: pitch sign flipped (sp=-1), yaw from CSV + d_yaw
    eff_pitch = -row.rot[1] + d_pitch
    eff_yaw = row.rot[2] + d_yaw
    d = pt - np.array(row.loc) * CM_TO_M
    fwd, right, up = axes(eff_pitch, eff_yaw, row.rot[0])
    pl = np.array([right @ d, -up @ d, fwd @ d])
    if pl[2] <= 0:
        return np.array([np.nan, np.nan])
    fx = s * focal * 1920 / sensor[0]
    fy = s * focal * 1080 / sensor[1]
    return np.array([fx * pl[0] / pl[2] + 960, fy * pl[1] / pl[2] + 540])


def fit_cam(cam_i):
    samples = ANCHORS[cam_i]
    def resid(p):
        d_yaw, d_pitch, s = p
        out = []
        for f, ua, va in samples:
            q = pixel(cam_i, f, gt_by[f], d_yaw, d_pitch, s)
            out.extend([q[0] - ua, q[1] - va])
        return np.array(out)
    res = least_squares(resid, [0.0, 0.0, 1.0])
    return res.x, np.abs(resid(res.x))


print("cam: d_yaw, d_pitch, focal_scale | residuals px | CSV yaw,pitch,roll -> eff")
for cam_i in (1, 2, 3, 4, 5):
    p, r = fit_cam(cam_i)
    row0 = cams[cam_i][0][ANCHORS[cam_i][0][0]]
    print(f"cam{cam_i}: d_yaw={p[0]:+.2f} d_pitch={p[1]:+.2f} s={p[2]:.3f} "
          f"max={r.max():.1f} mean={r.reshape(-1,2).mean(axis=1).mean():.1f} "
          f"csv=({row0.rot[2]:.1f},{row0.rot[1]:.1f},{row0.rot[0]:.1f})")
