"""Прокси-recall детектора по GT-пивоту: попадает ли пивот внутрь bbox.

2D-разметки объектов в датасете нет, поэтому precision/recall честно не
измерить. Но есть GT-пивот и калибровка: если детекция действительно
соответствует грузовику, проекция пивота должна попасть **внутрь** её bbox
(пивот — точка актора в его объёме). Это даёт:

* ``recall_proxy`` = доля кадров, где пивот в кадре **и** нашлась детекция,
  содержащая его проекцию (по лучшей по score детекции камеры);
* ``precision_proxy`` = доля всех детекций, содержащих проекцию пивотa;
* ``dist_px`` = расстояние от центра bbox до проекции пивота (медиана) —
  прямой измеритель того самого сдвига якоря, который даёт метры ошибки.

Скрипт **не запускает YOLO**: читает готовый CSV детекций
(``apps/detect_truck.py``). Поэтому сравнение моделей стоит секунды, если
детекции уже есть.

Запуск:
    python experiments/anchor_hitrate.py --detections out/truck/detections.csv
    python experiments/anchor_hitrate.py --detections out/truck/det_yolo26x.csv `
        --frames-csv out/truck/detections.csv   # те же кадры, что у базовой модели
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from multiview3d.camera import project_point
from multiview3d.truck_dataset import DEFAULT_REPAIRS, DEFAULT_ROOT, TruckDataset


def load_dets(path: str | Path) -> dict[tuple[str, int], list[dict]]:
    out: dict[tuple[str, int], list[dict]] = defaultdict(list)
    with Path(path).open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("x1") in (None, ""):
                continue
            out[(row["camera"], int(row["frame"]))].append({
                "score": float(row["score"]),
                "box": (float(row["x1"]), float(row["y1"]),
                        float(row["x2"]), float(row["y2"])),
                "center": (float(row["cx"]), float(row["cy"])),
            })
    return out


def inside(box, uv) -> bool:
    x1, y1, x2, y2 = box
    return x1 <= uv[0] <= x2 and y1 <= uv[1] <= y2


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Прокси-recall по GT-пивоту")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--repairs", default=str(DEFAULT_REPAIRS))
    ap.add_argument("--detections", required=True)
    ap.add_argument("--frames-csv", default=None,
                    help="ограничить кадры теми, что есть в другом CSV (честное сравнение)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    ds = TruckDataset.discover(args.root, repairs_path=args.repairs or None)
    gt = ds.ground_truth_by_frame()
    dets = load_dets(args.detections)
    if not dets:
        print(f"нет детекций в {args.detections}", file=sys.stderr)
        return 2

    frames = {f for _, f in dets}
    if args.frames_csv:
        frames &= {f for _, f in load_dets(args.frames_csv)}
    frames = sorted(frames)

    per_cam: dict[str, dict] = {}
    all_dist: list[float] = []
    for name in ds.names:
        n_visible = n_det = n_hit = n_boxes = n_boxes_hit = 0
        dist_hits: list[float] = []
        for f in frames:
            if f not in gt:
                continue
            cam = ds.provider.calibrations_at(f, nearest=True).get(name)
            if cam is None:
                continue
            try:
                uv = project_point(cam, gt[f])
            except ValueError:
                continue
            if not (0 <= uv[0] < cam.width and 0 <= uv[1] < cam.height):
                continue
            n_visible += 1
            boxes = dets.get((name, f), [])
            if not boxes:
                continue
            n_det += 1
            best = max(boxes, key=lambda b: b["score"])
            d = float(np.hypot(best["center"][0] - uv[0], best["center"][1] - uv[1]))
            dist_hits.append(d)
            if inside(best["box"], uv):
                n_hit += 1
            for b in boxes:
                n_boxes += 1
                n_boxes_hit += int(inside(b["box"], uv))
        per_cam[name] = {
            "frames_pivot_in_view": n_visible,
            "frames_with_detection": n_det,
            "frames_pivot_in_box": n_hit,
            "recall_proxy": round(n_hit / n_visible, 4) if n_visible else None,
            "precision_proxy": round(n_boxes_hit / n_boxes, 4) if n_boxes else None,
            "detections": n_boxes,
            "dist_center_to_pivot_px_median": (round(float(np.median(dist_hits)), 2)
                                               if dist_hits else None),
        }
        all_dist.extend(dist_hits)

    tot = {k: sum(v[k] for v in per_cam.values() if v[k])
           for k in ("frames_pivot_in_view", "frames_with_detection",
                     "frames_pivot_in_box", "detections")}
    n_boxes_hit = sum(round(v["precision_proxy"] * v["detections"])
                      for v in per_cam.values() if v["precision_proxy"] is not None)
    summary = {
        "detections_csv": str(args.detections).replace("\\", "/"),
        "frames_used": [frames[0], frames[-1], len(frames)] if frames else [],
        "totals": {
            **tot,
            "recall_proxy": round(tot["frames_pivot_in_box"]
                                  / tot["frames_pivot_in_view"], 4)
            if tot["frames_pivot_in_view"] else None,
            "precision_proxy": round(n_boxes_hit / tot["detections"], 4)
            if tot["detections"] else None,
            "dist_center_to_pivot_px_median": (round(float(np.median(all_dist)), 2)
                                               if all_dist else None),
        },
        "per_camera": per_cam,
        "note": ("recall_proxy — верхняя оценка полезности детекций: пивот внутри "
                 "bbox означает, что детекция соответствует объекту. 2D-разметки "
                 "нет, поэтому «настоящий» recall/precision не измеряется."),
    }

    t = summary["totals"]
    print(f"{args.detections}: кадров {len(frames)}, пивот в кадре "
          f"{t['frames_pivot_in_view']}, с детекцией {t['frames_with_detection']}, "
          f"пивот в bbox {t['frames_pivot_in_box']}")
    print(f"  recall_proxy={t['recall_proxy']}, precision_proxy={t['precision_proxy']}, "
          f"медиана |центр bbox − пивот|={t['dist_center_to_pivot_px_median']} px")
    for name, s in per_cam.items():
        print(f"  {name}: виден {s['frames_pivot_in_view']:4d}, детекций "
              f"{s['detections']:4d}, в bbox {s['frames_pivot_in_box']:4d}, "
              f"recall_proxy={s['recall_proxy']}, "
              f"dist_med={s['dist_center_to_pivot_px_median']} px")

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
