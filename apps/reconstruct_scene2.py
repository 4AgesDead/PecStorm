"""3D-реконструкция для задания 2 (scene2, 9 камер, без GT).

Запуск:
    .venv-rocm\\Scripts\\python.exe apps/reconstruct_scene2.py `
        --model yolov8x.pt --class-names truck `
        --start 350 --end 650 --step 2 `
        --out-csv out/scene2/est_scene2.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np

from multiview3d.detect import YoloDetector
from multiview3d.fusion import FusionConfig, MultiViewTracker
from multiview3d.scene2_dataset import Scene2Dataset
from multiview3d.types import Detection2D


def merge_detections(dets, iou_thr=0.5):
    """NMS + среднее боксов одного объекта от нескольких моделей."""
    if len(dets) <= 1:
        return dets
    dets = sorted(dets, key=lambda d: d.score, reverse=True)
    used = [False] * len(dets)
    out = []
    for i, d in enumerate(dets):
        if used[i]:
            continue
        group = [d]
        used[i] = True
        for j in range(i + 1, len(dets)):
            if used[j]:
                continue
            a, b = d.bbox, dets[j].bbox
            x1, y1 = max(a[0], b[0]), max(a[1], b[1])
            x2, y2 = min(a[2], b[2]), min(a[3], b[3])
            inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            ar = lambda q: max(0.0, q[2] - q[0]) * max(0.0, q[3] - q[1])
            iou = inter / (ar(a) + ar(b) - inter + 1e-9)
            if iou >= iou_thr:
                group.append(dets[j])
                used[j] = True
        w = np.array([max(float(g.score), 1e-6) for g in group])
        boxes = np.array([g.bbox for g in group], dtype=float)
        box = (boxes * w[:, None]).sum(axis=0) / w.sum()
        out.append(Detection2D(
            camera=d.camera, timestamp=d.timestamp,
            center=np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]),
            score=float(max(g.score for g in group)), bbox=tuple(box),
            label=d.label))
    return out


def pick_track(tracks):
    best, key_best = None, None
    for t in tracks:
        if not t.points:
            continue
        last = max(t.points, key=lambda p: (p.timestamp, p.status == "MEASURED"))
        key = (float(last.timestamp), last.status == "MEASURED")
        if key_best is None or key > key_best:
            best, key_best = t, key
    return best


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Scene2: 3D-траектория грузовика")
    ap.add_argument("--root", default="scene2")
    ap.add_argument("--model", default="yolov8x.pt")
    ap.add_argument("--models", default="",
                    help="ансамбль через запятую; если задан, --model игнорируется")
    ap.add_argument("--merge-iou", type=float, default=0.5)
    ap.add_argument("--class-names", default="truck")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default="0")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=1931)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--gate", type=float, default=10.0)
    ap.add_argument("--max-missing", type=int, default=5)
    ap.add_argument("--max-reprojection-px", type=float, default=50.0)
    ap.add_argument("--out-csv", default="out/scene2/est_scene2.csv")
    ap.add_argument("--out-json", default="out/scene2/est_scene2.json")
    args = ap.parse_args(argv)

    ds = Scene2Dataset.discover(args.root)
    names = ds.names
    model_paths = ([m.strip() for m in args.models.split(",") if m.strip()]
                   if args.models else [args.model])
    detectors = [YoloDetector(path, conf=args.conf, device=args.device,
                              imgsz=args.imgsz,
                              class_names=[c.strip() for c in args.class_names.split(",")
                                           if c.strip()])
                 for path in model_paths]
    print(f"детекторы: {model_paths}")
    cfg = FusionConfig(association_mode="single_target", min_views=2,
                       max_reprojection_px=args.max_reprojection_px,
                       gate_mahalanobis=args.gate, max_missing=args.max_missing)
    cams0 = ds.provider.calibrations_at(args.start, nearest=True)
    tracker = MultiViewTracker(names, [cams0[n] for n in names], cfg)
    caps = {c.name: cv2.VideoCapture(str(c.video)) for c in ds.cameras}
    for cap in caps.values():
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    rows = []
    n_meas = n_pred = 0
    last_pos = None
    last_frame = -10**9
    f = args.start
    while f <= args.end:
        imgs = []
        for name in names:
            ok, img = caps[name].read()
            imgs.append(img if ok else None)
        if any(im is None for im in imgs):
            break
        dets = []
        for name, img in zip(names, imgs):
            raw = []
            for det in detectors:
                raw.extend(det.detect(img))
            d = merge_detections(raw, args.merge_iou)
            for x in d:
                x.camera = name
                x.timestamp = f / ds.fps
            dets.append(d)
        cams = ds.provider.calibrations_at(f, nearest=True)
        tracker.set_cameras([cams[n] for n in names])
        res = tracker.step(f / ds.fps, dets, frame=f)
        track = None
        def measured_n(t):
            return sum(1 for p in t.points if p.status == "MEASURED")
        tracks = [t for t in res.tracks if t.points]
        for t in sorted(tracks, key=measured_n, reverse=True):
            p0 = t.points[-1]
            if p0.status != "MEASURED":
                continue
            far = (last_pos is not None and f - last_frame <= 30
                   and float(np.linalg.norm(np.asarray(p0.position) - last_pos)) > 5.0)
            if not far:
                track = t
                break
        if track is None:
            for t in sorted(tracks, key=measured_n, reverse=True):
                p0 = t.points[-1]
                if p0.status == "PREDICTED_ONLY" and (
                        last_pos is None or float(np.linalg.norm(
                            np.asarray(p0.position) - last_pos)) <= 20.0):
                    track = t
                    break
        row = {"frame": f, "timestamp": round(f / ds.fps, 6),
               "x": "", "y": "", "z": "", "vx": "", "vy": "", "vz": "",
               "status": "", "method": "", "n_views": 0, "views": "",
               "mean_reproj_px": "", "confidence": "", "track_id": ""}
        if track is not None:
            p = track.points[-1]
            if p.status == "MEASURED":
                last_pos = np.asarray(p.position, dtype=float)
                last_frame = f
            row.update({
                "x": round(float(p.position[0]), 6),
                "y": round(float(p.position[1]), 6),
                "z": round(float(p.position[2]), 6),
                "vx": round(float(p.velocity[0]), 6),
                "vy": round(float(p.velocity[1]), 6),
                "vz": round(float(p.velocity[2]), 6),
                "status": p.status, "method": p.method,
                "n_views": len(p.views or []),
                "views": "|".join(p.views or []),
                "mean_reproj_px": round(float(p.mean_reproj_px), 3)
                if np.isfinite(p.mean_reproj_px) else "",
                "confidence": round(float(p.confidence), 4)
                if np.isfinite(p.confidence) else "",
                "track_id": track.track_id,
            })
            if p.status == "MEASURED":
                n_meas += 1
            elif p.status == "PREDICTED_ONLY":
                n_pred += 1
        rows.append(row)
        if f % 50 == 0:
            print(f"  кадр {f}/{args.end}: track={None if track is None else track.track_id} "
                  f"status={row['status'] or '—'}")
        for _ in range(max(args.step - 1, 0)):
            for cap in caps.values():
                cap.read()
            f += 1
        f += 1

    for cap in caps.values():
        cap.release()
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    frames = len(rows)
    summary = {"root": args.root, "model": args.model, "frames": frames,
               "measured": n_meas, "predicted": n_pred,
               "coverage": round(n_meas / max(frames, 1), 3),
               "start": args.start, "end": args.end, "step": args.step,
               "gate": args.gate, "max_missing": args.max_missing}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    print(f"кадров: {frames}, MEASURED {n_meas}, PREDICTED {n_pred} "
          f"({summary['coverage'] * 100:.1f} %)")
    print(f"CSV: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
