"""Подбор конвенции координат UE->OpenCV: перебор sign-вариантов и проекция GT-пивота.

Эксперимент разработки: печатает пиксель пивота для 16 комбинаций знаков
(yaw/pitch/roll/right) на опорных кадрах; сопоставляется с видимым грузовиком.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from multiview3d.unreal_camera import parse_camera_csv, parse_track_csv, CM_TO_M  # noqa: E402

DATA = Path("scene1/zadania")
S = np.diag([1.0, -1.0, 1.0])          # UE -> правая мировая (на пиксели не влияет)
F2M = lambda xyz: np.array(xyz) * CM_TO_M


def axes_ue(pitch, yaw, roll, s_pitch=1.0, s_yaw=1.0, s_roll=1.0):
    sp, cp = np.sin(np.radians(s_pitch * pitch)), np.cos(np.radians(pitch))
    sy, cy = np.sin(np.radians(s_yaw * yaw)), np.cos(np.radians(yaw))
    sr, cr = np.sin(np.radians(s_roll * roll)), np.cos(np.radians(roll))
    fwd = np.array([cp * cy, cp * sy, sp])
    right = np.array([-sr * sp * cy + cr * sy, -sr * sp * sy - cr * cy, sr * cp])
    up = np.array([cr * sp * cy + sr * sy, cr * sp * sy - sr * cy, cr * cp])
    return fwd, right, up


def project(cam_row, loc, rot, focal, sensor, pt_m, sigma_right, s_pitch, s_yaw, s_roll,
            cx=960.0, cy=540.0):
    fx = focal * 1920 / sensor[0]
    fy = focal * 1080 / sensor[1]
    fwd, right, up = axes_ue(rot[1], rot[2], rot[0], s_pitch, s_yaw, s_roll)
    p_l = np.array([fwd @ loc, sigma_right * (right @ loc), -up @ loc])
    if p_l[2] <= 1e-9:
        return None
    return (fx * p_l[0] / p_l[2] + cx, fy * p_l[1] / p_l[2] + cy, float(np.linalg.norm(loc)))


def main() -> int:
    cams = {}
    for i in range(1, 6):
        p = DATA / f"xyzpowcameta/object_transform_CineCameraActor{i}.csv"
        pr = parse_camera_csv(p)
        cams[f"cam{i}"] = (pr.rows[0], pr.focal_mm, pr.sensor_mm)
    gt = parse_track_csv(DATA / "xyzpowcameta/object_transform_SM_AA_track.csv")
    gt_by = {r.frame: np.array(r.loc) * CM_TO_M for r in gt.rows}

    probes = [(1, 0), (2, 300), (3, 300), (4, 900), (5, 750)]
    print("Пивот (относительно центра камеры):")
    header = " | ".join(f"c{c}f{f}" for c, f in probes)
    for sr in (1.0, -1.0):
        for sp_ in (1.0, -1.0):
            for sy_ in (1.0, -1.0):
                for so_ in (1.0, -1.0):
                    cells = []
                    for ci, f in probes:
                        row, focal, sensor = cams[f"cam{ci}"]
                        c = np.array(row.loc) * CM_TO_M
                        pt = gt_by.get(f)
                        d = pt - c
                        fwd, right, up = axes_ue(row.rot[1], row.rot[2], row.rot[0], sp_, sy_, so_)
                        pl = np.array([sr * (right @ d), -up @ d, fwd @ d])
                        if pl[2] <= 1e-9:
                            cells.append("behind")
                            continue
                        fx = focal * 1920 / sensor[0]
                        fy = focal * 1080 / sensor[1]
                        u = fx * pl[0] / pl[2] + 960
                        v = fy * pl[1] / pl[2] + 540
                        cells.append(f"{u:.0f},{v:.0f}({pl[2]:.0f}m)")
                    print(f"sr={sr:+.0f} sp={sp_:+.0f} sy={sy_:+.0f} so={so_:+.0f} | {header}")
                    print(f"    " + " | ".join(cells))
    # сырые кадры для ручной сверки
    out = Path("out/truck_sanity")
    out.mkdir(parents=True, exist_ok=True)
    for ci, f in probes:
        cap = cv2.VideoCapture(str(DATA / f"camera/Way1.FinalImage.CineCameraActor{ci}.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        cap.release()
        if ok:
            cv2.imwrite(str(out / f"raw_cam{ci}_frame{f}.png"), img)
            print(f"raw: out/truck_sanity/raw_cam{ci}_frame{f}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
