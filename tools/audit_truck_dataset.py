"""Аудит датасета «Пеленга»: видео, CSV камер, GT-траектория -> manifest + отчёт.

Запуск:  python tools/audit_truck_dataset.py [--root huniashahatona/zadania]

Создаёт (не изменяя входные файлы):
  datasets/truck/dataset_manifest.json
  datasets/truck/ground_truth_repairs.json  (гипотезы восстановлений, verified=false)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from multiview3d.unreal_camera import (  # noqa: E402
    parse_camera_csv, parse_track_csv, propose_repairs, ue_rot_to_matrix, M_S,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe_video(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"path": str(path), "error": "не открывается"}
    info = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path)[:16],
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "fourcc": str(cap.get(cv2.CAP_PROP_FOURCC)),
        "frames_property": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    decodable = 0
    ok0 = cap.grab()
    if ok0:
        decodable = 1
        while cap.grab():
            decodable += 1
    info["frames_decodable"] = decodable
    cap.release()
    return info


def analyze_camera(name: str, parsed) -> dict:
    rows = parsed.rows
    locs = np.array([r.loc for r in rows])
    rots = np.array([r.rot for r in rows])
    static = bool(np.allclose(locs, locs[0]) and np.allclose(rots, rots[0]))
    out = {
        "frames": [rows[0].frame, rows[-1].frame],
        "n_rows": len(rows),
        "static": static,
        "focal_mm": parsed.focal_mm,
        "sensor_mm": list(parsed.sensor_mm) if parsed.sensor_mm else None,
        "invalid_cells": parsed.invalid_cells,
    }
    if not static:
        yaw = rots[:, 2]
        d = np.diff(yaw)
        jumps = int(np.sum(np.abs(d) > 179.0))
        # непрерывный unwrap и шаг матрицы поворота (градусы)
        yaw_unwrapped = np.unwrap(np.radians(yaw))
        total = float(np.degrees(yaw_unwrapped[-1] - yaw_unwrapped[0]))
        max_step = 0.0
        for i in range(len(rows) - 1):
            if rows[i + 1].frame != rows[i].frame + 1:
                continue
            r1 = ue_rot_to_matrix(*rows[i].rot)
            r2 = ue_rot_to_matrix(*rows[i + 1].rot)
            ang = float(np.degrees(np.arccos(np.clip((np.trace(r1.T @ r2) - 1) / 2, -1, 1))))
            max_step = max(max_step, ang)
        out["yaw_raw_min"] = float(yaw.min())
        out["yaw_raw_max"] = float(yaw.max())
        out["yaw_wrap_jumps"] = jumps
        out["yaw_total_unwrapped_deg"] = total
        out["max_rotation_step_deg_per_frame"] = max_step
        out["rot_columns_constant"] = {
            "roll": bool(np.allclose(rots[:, 0], rots[0, 0])),
            "pitch": bool(np.allclose(rots[:, 1], rots[0, 1])),
            "yaw": bool(np.allclose(rots[:, 2], rots[0, 2])),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="huniashahatona/zadania")
    args = ap.parse_args()
    root = Path(args.root)
    cam_dir, gt_dir = root / "camera", root / "xyzpowcameta"
    out_dir = Path("datasets/truck")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== ВИДЕО ===")
    videos = {}
    for vp in sorted(cam_dir.glob("*.mp4")):
        info = probe_video(vp)
        videos[vp.name] = info
        print(f"{vp.name}: {info.get('width')}x{info.get('height')} "
              f"fps={info.get('fps')} fourcc={info.get('fourcc')} "
              f"frames(prop)={info.get('frames_property')} "
              f"frames(decodable)={info.get('frames_decodable')}")

    print("\n=== CSV КАМЕР ===")
    cameras = {}
    for cp in sorted(gt_dir.glob("object_transform_CineCameraActor*.csv")):
        parsed = parse_camera_csv(cp)
        a = analyze_camera(cp.name, parsed)
        cameras[cp.name] = a
        print(f"{cp.name}: строк={a['n_rows']} кадры={a['frames']} static={a['static']} "
              f"focal={a['focal_mm']}mm sensor={a['sensor_mm']} "
              f"битых ячеек={len(a['invalid_cells'])}")
        if not a["static"]:
            print(f"   yaw raw [{a['yaw_raw_min']:.4f}, {a['yaw_raw_max']:.4f}], "
                  f"wrap-скачков={a['yaw_wrap_jumps']}, итог unwrap={a['yaw_total_unwrapped_deg']:.3f}°, "
                  f"макс шаг поворота/кадр={a['max_rotation_step_deg_per_frame']:.4f}°")

    print("\n=== GT ЦЕЛИ ===")
    gt_path = gt_dir / "object_transform_SM_AA_track.csv"
    gt = parse_track_csv(gt_path)
    arr = np.array([r.loc for r in gt.rows])
    print(f"строк={len(gt.rows)} кадры=[{gt.rows[0].frame}, {gt.rows[-1].frame}]")
    print(f"X [{arr[:, 0].min():.4f}, {arr[:, 0].max():.4f}] "
          f"Y [{arr[:, 1].min():.4f}, {arr[:, 1].max():.4f}] "
          f"Z [{arr[:, 2].min():.4f}, {arr[:, 2].max():.4f}] (см, UE)")
    for c in gt.invalid_cells:
        print(f"  БИТАЯ ЯЧЕЙКА: строка {c['line']} frame={c['frame']} "
              f"column={c['column']} raw={c['raw_value']!r}")
        f = c["frame"]
        prev = next((r for r in reversed(gt.rows) if r.frame < f), None)
        nxt = next((r for r in gt.rows if r.frame > f), None)
        col = {"Loc_X": 0, "Loc_Y": 1, "Loc_Z": 2}[c["column"]]
        if prev and nxt:
            print(f"    соседи: frame {prev.frame} {c['column']}={prev.loc[col]:.4f}, "
                  f"frame {nxt.frame} {c['column']}={nxt.loc[col]:.4f}")
    repairs = propose_repairs(gt.invalid_cells)
    if repairs:
        rep_path = out_dir / "ground_truth_repairs.json"
        rep_path.write_text(json.dumps({
            "source_file": str(gt_path),
            "accept_unverified": False,
            "note": "Гипотезы восстановления Excel date coercion. "
                    "Требуется подтверждение организаторами.",
            "repairs": repairs,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        for r in repairs:
            print(f"  предложение: frame={r['frame']} {r['column']}: "
                  f"{r['source_raw_value']!r} -> {r['proposed_value']} (не подтверждено)")

    manifest = {
        "generated_by": "tools/audit_truck_dataset.py",
        "units_assumption": "Loc_* в сантиметрах Unreal; внутренняя система — метры (x0.01)",
        "coordinate_assumption": "правая UE_ue: X=X_ue, Y=-Y_ue, Z=Z_ue",
        "intrinsics_assumption": "pinhole из focal/sensor, principal point в центре, дисторсия=0 (не подтверждено)",
        "videos": videos,
        "cameras": cameras,
        "ground_truth": {
            "path": str(gt_path),
            "sha256_16": sha256(gt_path)[:16],
            "n_rows": len(gt.rows),
            "frames": [gt.rows[0].frame, gt.rows[-1].frame],
            "invalid_cells": gt.invalid_cells,
            "repairs_file": "datasets/truck/ground_truth_repairs.json" if repairs else None,
        },
    }
    mf = out_dir / "dataset_manifest.json"
    mf.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nmanifest: {mf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
