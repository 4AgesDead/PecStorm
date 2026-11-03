"""Фит постоянного 3D-сдвига якоря по ML-детекциям (bbox center) и GT.

Задача: детектор возвращает центр bbox, а ground truth — позиция ``pivot``
актора. Это разные 3D-точки, поэтому триангуляция по bbox-центрам даёт
систематическое смещение. Модель: ``pivot_gt[f] + delta`` проецируется в
центр bbox на каждой камере; ``delta`` (3 числа, метры) — постоянная поправка.

Запуск:
    python tools/anchor_fit_yolo.py --train-start 0 --train-end 500 --step 25
        --holdout-start 600 --holdout-end 1000
        --out datasets/truck/anchor_offset.json

Правила:
* ``delta`` оценивается **только** на обучающих кадрах (``--train-*``);
* проверка — на отложенных кадрах (``--holdout-*``), они в фите не участвуют;
* GT не используется в самом конвейере: он нужен только для калибровки этого
  постоянного сдвига, что прямо документируется в docs/TRUCK_DATASET.md;
* исходные файлы датасета не изменяются.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from scipy.optimize import least_squares

from multiview3d.camera import project_point
from multiview3d.detect import build_detector
from multiview3d.truck_dataset import (
    DEFAULT_REPAIRS,
    DEFAULT_ROOT,
    TruckDataset,
    TruckVideoSource,
)


def collect(ds: TruckDataset, detector, frames: list[int],
            anchor: str = "center") -> list[dict]:
    """Лучшая детекция каждой камеры на кадре + GT: список наблюдений.

    ``anchor="center"`` — центр bbox; ``anchor="bottom_center"`` — середина
    нижней границы bbox (точка контакта с дорогой). Детекции, обрезанные снизу
    кадром, для ``bottom_center`` отбрасываются: их нижняя граница не
    соответствует реальной точке контакта.
    """
    gt = {int(f): p for f, p in zip(ds.ground_truth().frames, ds.ground_truth().positions_m)}
    todo = [f for f in frames if f in gt]
    if not todo:
        return []
    sources = {c.name: TruckVideoSource(c, start=todo[0], end=todo[-1], fps=ds.fps)
               for c in ds.cameras}
    obs: list[dict] = []
    try:
        for frame in todo:
            for name, src in sources.items():
                while src.frame_index < frame:
                    if src.read() is None:
                        raise RuntimeError(f"{name}: не дочитан до кадра {frame}")
                item = src.read()
                if item is None:
                    continue
                _, image, _ = item
                height = image.shape[0]
                dets = [d for d in detector.detect(image) if d.bbox is not None]
                if not dets:
                    continue
                best = max(dets, key=lambda d: d.score)
                x1, y1, x2, y2 = best.bbox
                if anchor == "bottom_center":
                    if y2 >= height - 2:
                        continue  # нижняя граница обрезана кадром
                    center = np.array([(x1 + x2) / 2.0, y2], dtype=float)
                else:
                    center = np.asarray(best.center, dtype=float)
                obs.append({"camera": name, "frame": frame,
                            "center": center, "score": float(best.score),
                            "gt": np.asarray(gt[frame], float)})
    finally:
        for src in sources.values():
            src.close()
    return obs


def collect_from_detections(obs_map: dict, ds: TruckDataset, frames: list[int],
                            anchor: str = "center") -> list[dict]:
    """Наблюдения из готовых 2D-детекций (CSV ``apps/detect_truck.py``).

    Экономит повторный прогон детектора: фит использует те же детекции, что и
    конвейер. Выбирается лучшая по score детекция камеры на кадре.
    """
    gt = {int(f): p for f, p in zip(ds.ground_truth().frames, ds.ground_truth().positions_m)}
    height = ds.height
    out: list[dict] = []
    for frame in frames:
        if frame not in gt:
            continue
        for name, dets in (obs_map.get(frame) or {}).items():
            dets = [d for d in dets if d.bbox is not None]
            if not dets:
                continue
            best = max(dets, key=lambda d: d.score)
            x1, y1, x2, y2 = best.bbox
            if anchor == "bottom_center":
                if y2 >= height - 2:
                    continue
                center = np.array([(x1 + x2) / 2.0, y2], dtype=float)
            else:
                center = np.asarray(best.center, dtype=float)
            out.append({"camera": name, "frame": int(frame), "center": center,
                        "score": float(best.score), "gt": np.asarray(gt[frame], float)})
    return out


def residuals(obs: list[dict], ds: TruckDataset, offset: np.ndarray) -> np.ndarray:
    """Пиксельные невязки ``project(cam_f, gt+offset) - center`` по наблюдениям."""
    cams: dict[tuple[str, int], object] = {}
    out = []
    for o in obs:
        key = (o["camera"], o["frame"])
        cam = cams.get(key)
        if cam is None:
            cam = ds.calibration(o["camera"], o["frame"])
            cams[key] = cam
        try:
            uv = project_point(cam, o["gt"] + offset)
        except ValueError:
            out.extend([1e3, 1e3])
            continue
        if not np.all(np.isfinite(uv)):
            out.extend([1e3, 1e3])
            continue
        out.extend([float(uv[0] - o["center"][0]), float(uv[1] - o["center"][1])])
    return np.asarray(out, dtype=float)


def summarize(r: np.ndarray) -> dict:
    if r.size == 0:
        return {"n": 0}
    d = np.linalg.norm(r.reshape(-1, 2), axis=1)
    return {"n_observations": int(d.size),
            "px_mean": round(float(np.mean(d)), 2),
            "px_median": round(float(np.median(d)), 2),
            "px_p90": round(float(np.percentile(d, 90)), 2),
            "px_max": round(float(np.max(d)), 2)}


def cross_validate_by_camera(obs: list[dict], ds: TruckDataset) -> dict:
    """Leave-one-camera-out: подгонка без камеры, проверка на ней.

    Показывает, переносится ли постоянный 3D-сдвиг на камеру, данных которой
    не было в фите; это честнее, чем residual на тех же наблюдениях.
    """
    out = {}
    cameras = sorted({o["camera"] for o in obs})
    for cam in cameras:
        train = [o for o in obs if o["camera"] != cam]
        test = [o for o in obs if o["camera"] == cam]
        if len(train) < 6 or not test:
            out[cam] = {"train_observations": len(train),
                        "test_observations": len(test), "skipped": True}
            continue
        res = least_squares(lambda d: residuals(train, ds, d), x0=np.zeros(3),
                            loss="soft_l1", f_scale=20.0, max_nfev=200)
        out[cam] = {
            "train_observations": len(train),
            "test_observations": len(test),
            "offset_m": [round(float(-v), 4) for v in res.x],
            "delta_pivot_to_bbox_m": [round(float(v), 4) for v in res.x],
            "test_px": summarize(residuals(test, ds, np.asarray(res.x))),
        }
    return out


def per_camera(obs: list[dict], ds: TruckDataset, offset: np.ndarray) -> dict:
    out = {}
    for name in ds.names:
        r = residuals([o for o in obs if o["camera"] == name], ds, offset)
        out[name] = summarize(r)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Фит 3D-сдвига якоря по YOLO-детекциям")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--repairs", default=str(DEFAULT_REPAIRS))
    ap.add_argument("--train-start", type=int, default=0)
    ap.add_argument("--train-end", type=int, default=500)
    ap.add_argument("--holdout-start", type=int, default=600)
    ap.add_argument("--holdout-end", type=int, default=1000)
    ap.add_argument("--step", type=int, default=25)
    ap.add_argument("--model", default="yolov8x.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--anchor", default="center", choices=["center", "bottom_center"],
                    help="2D-якорь детекции: центр bbox или середина нижней границы")
    ap.add_argument("--detections", default=None,
                    help="готовый CSV детекций (не запускать YOLO повторно)")
    ap.add_argument("--out", default="datasets/truck/anchor_offset.json")
    ap.add_argument("--report", default="out/anchor_fit_yolo.json")
    args = ap.parse_args(argv)

    ds = TruckDataset.discover(args.root, repairs_path=args.repairs or None)
    train = list(range(args.train_start, args.train_end + 1, args.step))
    holdout = list(range(args.holdout_start, args.holdout_end + 1, args.step))

    print(f"Сбор детекций: train {train[0]}..{train[-1]} шаг {args.step}, "
          f"holdout {holdout[0]}..{holdout[-1]}")
    if args.detections:
        from multiview3d.truck_pipeline import load_detections
        cached = load_detections(args.detections)
        print(f"источник: {args.detections} ({len(cached)} кадров)")
        obs_train = collect_from_detections(cached, ds, train, anchor=args.anchor)
        obs_hold = collect_from_detections(cached, ds, holdout, anchor=args.anchor)
    else:
        detector = build_detector({"kind": "yolo", "model": args.model, "conf": args.conf,
                                   "imgsz": args.imgsz,
                                   "class_names": ["truck", "car", "bus", "train"]})
        obs_train = collect(ds, detector, train, anchor=args.anchor)
        obs_hold = collect(ds, detector, holdout, anchor=args.anchor)
    print(f"наблюдений: train {len(obs_train)}, holdout {len(obs_hold)}")
    if len(obs_train) < 6:
        print("слишком мало наблюдений для фита трёх параметров", file=sys.stderr)
        return 2

    zero = np.zeros(3)
    res = least_squares(lambda d: residuals(obs_train, ds, d), x0=zero,
                        loss="soft_l1", f_scale=20.0, x_scale=1.0, max_nfev=200)
    offset = np.asarray(res.x, dtype=float)
    print(f"delta (pivot->bbox) = ({offset[0]:+.3f}, {offset[1]:+.3f}, "
          f"{offset[2]:+.3f}) м, сдвиг для оценки = "
          f"({-offset[0]:+.3f}, {-offset[1]:+.3f}, {-offset[2]:+.3f}) м, "
          f"cost={res.cost:.1f}, status={res.status}")

    before = summarize(residuals(obs_train, ds, zero))
    after = summarize(residuals(obs_train, ds, offset))
    print(f"train: до {before['px_mean']:.2f} px (медиана {before['px_median']:.2f}) -> "
          f"после {after['px_mean']:.2f} px (медиана {after['px_median']:.2f})")
    hold = summarize(residuals(obs_hold, ds, offset))
    hold_before = summarize(residuals(obs_hold, ds, zero))
    if hold["n_observations"]:
        print(f"holdout: до {hold_before['px_mean']:.2f} px -> после {hold['px_mean']:.2f} px "
              f"(медиана {hold['px_median']:.2f}, p90 {hold['px_p90']:.2f})")
    per_cam = per_camera(obs_hold, ds, offset)
    for name, s in per_cam.items():
        if s.get("n_observations"):
            print(f"  holdout {name}: n={s['n_observations']}, "
                  f"mean={s['px_mean']} px, median={s['px_median']} px")

    loco = cross_validate_by_camera(obs_train + obs_hold, ds)
    for name, s in loco.items():
        if s.get("skipped"):
            continue
        print(f"  leave-one-camera-out {name}: train n={s['train_observations']}, "
              f"test mean={s['test_px']['px_mean']} px "
              f"(offset {s['offset_m']})")

    doc = {
        "offset_m": [float(-v) for v in offset],
        "delta_pivot_to_bbox_m": [float(v) for v in offset],
        "method": ("least_squares soft_l1: project(pivot_gt + delta) ≈ bbox_center; "
                   "offset_m = -delta — сдвиг, который прибавляется к "
                   "триангулированной точке, чтобы перейти к pivot"),
        "anchor": args.anchor,
        "detector": {"model": args.model, "conf": args.conf, "imgsz": args.imgsz,
                     "class_names": ["truck", "car", "bus", "train"]},
        "train": {"frames": [train[0], train[-1], args.step], **after},
        "train_before": before,
        "holdout": {"frames": [holdout[0], holdout[-1], args.step], **hold},
        "holdout_before": hold_before,
        "per_camera_holdout": per_cam,
        "leave_one_camera_out": loco,
        "note": ("Постоянный сдвиг оценён на обучающих кадрах; на проверочных не "
                 "подбирался. Это калибровка якоря (bbox center != actor pivot), "
                 "а не подгонка траектории. Остаточные десятки пикселей означают "
                 "неточность intrinsics/extrinsics и неизвестный anchor."),
    }
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    rp = Path(args.report)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"записано: {p} и {rp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
