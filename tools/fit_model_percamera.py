"""Подгонка попиксельного сдвига проекции модели отдельно по камерам.

Причина: курс в GT отсутствует, он считается по разности позиций; плюс сдвиг
модели относительно pivot. В результате проекция систематически смещена, и
смещение своё у каждой камеры (cam1 IoU 0.74, cam5 0.40). Здесь для каждой
камеры подбирается сдвиг (dx, dy) в пикселях, максимизирующий IoU с bbox на
train-кадрах. Результат дописывается в model_alignment.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from multiview3d.truck_dataset import DEFAULT_REPAIRS, DEFAULT_ROOT, TruckDataset
from multiview3d.truck_pipeline import load_detections
from tools.fit_truck_model import corners_world, fbx_size, heading_from_gt, iou, project_box


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Попиксельный сдвиг модели по камерам")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--repairs", default=str(DEFAULT_REPAIRS))
    ap.add_argument("--model", default="models/CargoTruck.fbx")
    ap.add_argument("--alignment", default="datasets/truck/model_alignment.json")
    ap.add_argument("--detections", default="out/truck/detections.csv")
    ap.add_argument("--train-end", type=int, default=700)
    ap.add_argument("--limit", type=int, default=120, help="наблюдений на камеру")
    args = ap.parse_args(argv)

    align = json.loads(Path(args.alignment).read_text(encoding="utf-8"))
    size = fbx_size(args.model) * float(align["scale"])
    offset = np.asarray(align["offset_m"], dtype=float)
    dyaw = np.deg2rad(float(align["dyaw_deg"]))
    beta = float(align.get("pivot_frac", 0.0))
    up, fwd = align["axes"]["up"], align["axes"]["forward"]
    sign = float(align["axes"]["forward_sign"])

    ds = TruckDataset.discover(args.root, repairs_path=args.repairs or None)
    gt = ds.ground_truth_by_frame()
    yaw_gt = heading_from_gt(gt)
    dets = load_detections(args.detections)

    per_cam: dict[str, list] = {n: [] for n in ds.names}
    for f in sorted(dets):
        if f > args.train_end or f not in gt:
            continue
        cams = ds.provider.calibrations_at(f, nearest=True)
        for name, items in dets[f].items():
            items = [d for d in items if d.bbox is not None]
            if not items or name not in cams:
                continue
            b = max(items, key=lambda d: d.score)
            per_cam[name].append((f, cams[name], tuple(float(v) for v in b.bbox)))

    result = {}
    for name, obs in per_cam.items():
        if len(obs) > args.limit:
            idx = np.linspace(0, len(obs) - 1, args.limit).astype(int)
            obs = [obs[i] for i in idx]
        if not obs:
            result[name] = {"shift_px": [0.0, 0.0], "iou": None, "n": 0}
            continue

        boxes = []
        for f, cam, box in obs:
            pts = corners_world(size, np.asarray(gt[f], dtype=float) + offset,
                                yaw_gt[f] + dyaw, up, fwd, sign, pivot_frac=beta)
            pb = project_box(cam, pts)
            if pb is not None:
                boxes.append((pb, box))
        if not boxes:
            result[name] = {"shift_px": [0.0, 0.0], "iou": None, "n": 0}
            continue

        def score(dx, dy):
            return float(np.mean(
                [iou((pb[0] + dx, pb[1] + dy, pb[2] + dx, pb[3] + dy), box)
                 for pb, box in boxes]))

        best = (score(0.0, 0.0), 0.0, 0.0)
        for dx in range(-150, 151, 15):
            for dy in range(-150, 151, 15):
                s = score(float(dx), float(dy))
                if s > best[0]:
                    best = (s, float(dx), float(dy))
        for dx in np.arange(best[1] - 14, best[1] + 15, 3.5):
            for dy in np.arange(best[2] - 14, best[2] + 15, 3.5):
                s = score(float(dx), float(dy))
                if s > best[0]:
                    best = (s, float(dx), float(dy))
        result[name] = {"shift_px": [round(best[1], 2), round(best[2], 2)],
                        "iou": round(best[0], 4), "n": len(boxes)}
        print(f"{name}: сдвиг ({best[1]:+.1f}, {best[2]:+.1f}) px -> IoU {best[0]:.3f} (n={len(boxes)})")

    align["camera_shift_px"] = {k: v["shift_px"] for k, v in result.items()}
    align["per_camera_fit"] = result
    Path(args.alignment).write_text(json.dumps(align, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
    print(f"записано: {args.alignment}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
