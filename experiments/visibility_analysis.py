"""Анализ видимости: сколько камер видят GT-пивот цели и на каких интервалах.

Запуск:
    python experiments/visibility_analysis.py [--step 5] [--out out/experiments]

Зачем: определяет, на каких кадрах вообще возможно измерение 3D (>=2 вида).
Без этого доля «измеренных» кадров не интерпретируется (§7.6 брифа).

Внимание: видимость считается по **проекции GT-пивота в кадр**, т.е. это
верхняя оценка: реальный детектор может не найти объект (малый размер,
окклюзия), поэтому итоговое покрытие берётся из ``apps/evaluate_truck.py``.
GT используется только для анализа, не для оценок конвейера.
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

from multiview3d.camera import project_point
from multiview3d.truck_dataset import TruckDataset


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Видимость цели по камерам")
    ap.add_argument("--root", default="huniashahatona/zadania")
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--out", default="out/experiments")
    args = ap.parse_args(argv)

    ds = TruckDataset.discover(args.root)
    gt = ds.ground_truth_by_frame()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_cam = {n: 0 for n in ds.names}
    hist: dict[int, int] = {}
    frames = []
    for f in range(0, ds.frame_count(), args.step):
        if f not in gt:
            continue
        cams = ds.provider.calibrations_at(f, nearest=True)
        inside = []
        for name, cam in cams.items():
            try:
                uv = project_point(cam, gt[f])
            except ValueError:
                continue
            if 0 <= uv[0] < cam.width and 0 <= uv[1] < cam.height:
                inside.append(name)
                per_cam[name] += 1
        hist[len(inside)] = hist.get(len(inside), 0) + 1
        frames.append({"frame": f, "cameras": sorted(inside)})

    total = len(frames)
    doc = {
        "step": args.step,
        "frames_analyzed": total,
        "frames_with_gt": len(gt),
        "cameras_per_frame_histogram": {str(k): v for k, v in sorted(hist.items())},
        "frames_with_ge2_cameras": sum(v for k, v in hist.items() if k >= 2),
        "share_ge2_cameras": (sum(v for k, v in hist.items() if k >= 2) / total) if total else None,
        "per_camera_frames_visible": per_cam,
        "per_camera_share": {k: v / total for k, v in per_cam.items()} if total else {},
        "frames": frames,
    }
    (out_dir / "visibility.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"Кадров проанализировано: {total} (шаг {args.step}), GT-кадров: {len(gt)}")
    print("Распределение числа видимых камер на кадре:")
    for k in sorted(hist):
        print(f"  {k} камер: {hist[k]} кадров ({100.0 * hist[k] / total:.1f} %)")
    print(f"Кадров с >=2 видами: {doc['frames_with_ge2_cameras']} "
          f"({100.0 * (doc['share_ge2_cameras'] or 0):.1f} %)")
    print("Доля кадров, где пивот в кадре камеры:")
    for name in ds.names:
        print(f"  {name}: {100.0 * per_cam[name] / total:.1f} %")
    print(f"JSON: {out_dir / 'visibility.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
