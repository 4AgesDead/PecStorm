"""Проверка конвенций UE->OpenCV по вручную отмеченным рамкам грузовика (full-res)."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")
from multiview3d.unreal_camera import parse_camera_csv, parse_track_csv, CM_TO_M

DATA = Path("huniashahatona/zadania")
M_LOCAL = np.array([[0., 1, 0], [0, 0, -1], [1, 0, 0]])
S_FLIP = np.diag([1., -1, 1])

# вручную отмеченные bbox (u1,u2,v1,v2) в full-res 1920x1080, по просмотренным кадрам
BOXES = {
    (1, 0):   (95, 350, 415, 505),
    (2, 300): (770, 900, 450, 540),
    (2, 450): (730, 910, 430, 530),
    (4, 900): (400, 580, 585, 660),
    (5, 750): (1080, 1540, 380, 700),
    (3, 450): (1680, 1920, 480, 580),
}

cams = {}
for i in range(1, 6):
    pr = parse_camera_csv(DATA / f"xyzpowcameta/object_transform_CineCameraActor{i}.csv")
    cams[i] = ({r.frame: r for r in pr.rows}, pr.focal_mm, pr.sensor_mm)
gt = parse_track_csv(DATA / "xyzpowcameta/object_transform_SM_AA_track.csv")
gt_by = {r.frame: np.array(r.loc) * CM_TO_M for r in gt.rows}


def axes(pitch, yaw, roll, sp=1.0, sy=1.0, so=1.0):
    s_p, c_p = np.sin(np.radians(sp * pitch)), np.cos(np.radians(pitch))
    s_y, c_y = np.sin(np.radians(sy * yaw)), np.cos(np.radians(yaw))
    s_o, c_o = np.sin(np.radians(so * roll)), np.cos(np.radians(roll))
    fwd = np.array([c_p * c_y, c_p * s_y, s_p])
    right = np.array([-s_o * s_p * c_y + c_o * s_y, -s_o * s_p * s_y - c_o * c_y, s_o * c_p])
    up = np.array([c_o * s_p * c_y + s_o * s_y, c_o * s_p * s_y - s_o * c_y, c_o * c_p])
    return fwd, right, up


def predict(cam_i, frame, sr, sp, sy, so):
    rows, focal, sensor = cams[cam_i]
    row = rows[frame]
    pt = gt_by[frame]
    d = np.array(row.loc) * CM_TO_M - pt   # camera -> pivot, UE метры (без флипа)
    fwd, right, up = axes(row.rot[1], row.rot[2], row.rot[0], sp, sy, so)
    pl = np.array([sr * (right @ -d), -up @ -d, fwd @ -d])  # d_cam = pivot - cam = -d
    if pl[2] <= 0:
        return None
    fx = focal * 1920 / sensor[0]; fy = focal * 1080 / sensor[1]
    return (fx * pl[0] / pl[2] + 960, fy * pl[1] / pl[2] + 540, pl[2])


print("sr sp sy so | " + " | ".join(f"c{c}f{f}" for c, f in BOXES))
for sr in (1., -1.):
    for sp in (1., -1.):
        for sy in (1., -1.):
            hits, misses = 0, 0
            cells = []
            for (ci, f), (u1, u2, v1, v2) in BOXES.items():
                p = predict(ci, f, sr, sp, sy, 1.0)
                if p is None:
                    cells.append("behind")
                    misses += 1
                    continue
                inside = u1 - 60 <= p[0] <= u2 + 60 and v1 - 60 <= p[1] <= v2 + 60
                hits += inside
                misses += not inside
                cells.append(f"{p[0]:.0f},{p[1]:.0f}{'✓' if inside else ''}")
            print(f"{sr:+.0f} {sp:+.0f} {sy:+.0f} +1 | hits {hits}/{len(BOXES)}: " + " | ".join(cells))
