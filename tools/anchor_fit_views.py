"""Калибровка 3D-сдвига якоря **на набор видов** и честная проверка переноса.

Зачем: детектор даёт центр bbox, GT — pivot актора. Смещение «bbox center ↔
pivot» есть в каждом виде, но **у пары видов** оно своё: триангуляция точек,
которые в разных видах соответствуют разным частям грузовика, даёт точку,
смещённую по-разному для разных пар. Поэтому один общий 3D-сдвиг
(``anchor_offset_m``) не описывает все пары, а набор пар в train и holdout
может вообще не пересекаться.

Модель: ``estimate_corrected = estimate + offset[views]``, где ``views`` —
канонический набор камер (``"cam3|cam5"``). Оценка ``offset`` — покоординатная
медиана вектора ``estimate - gt`` по кадрам этого набора видов.

Проверка переноса (главное):
* ``parity`` — фит на чётных кадрах набора, проверка на нечётных и наоборот;
* ``time`` — фит на кадрах ``--fit-start..--fit-end``, проверка на
  ``--test-start..--test-end`` (для этой цели).
Оба числа печатаются и пишутся в JSON: если перенос не подтверждается,
сдвиг использовать нельзя, даже если in-sample MAE красивая.

Запуск:
    python tools/anchor_fit_views.py --detections out/truck/detections.csv `
        --out datasets/truck/anchor_offset_views.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from multiview3d.fusion import FusionConfig, canonical_view_key
from multiview3d.truck_dataset import TruckDataset
from multiview3d.truck_pipeline import (
    estimate_points,
    load_detections,
    run_reconstruction,
)
from multiview3d.types import MEASURED


def residuals(ds: TruckDataset, dets: dict, frames: list[int],
              fusion: FusionConfig) -> list[dict]:
    """Точки MEASURED без якорей: `{frame, views, res (est - gt)}`."""
    res = run_reconstruction(ds, dets, fusion=fusion, frames=frames)
    gt = ds.ground_truth_by_frame()
    out: list[dict] = []
    for p in estimate_points(res):
        if p.status != MEASURED or p.frame is None or p.frame not in gt:
            continue
        key = canonical_view_key(p.views)
        out.append({
            "frame": int(p.frame),
            "views": key,
            "res": np.asarray(p.position, dtype=float) - np.asarray(gt[p.frame], dtype=float),
            "err_m": float(np.linalg.norm(np.asarray(p.position, dtype=float)
                                          - np.asarray(gt[p.frame], dtype=float))),
        })
    return out


def fit(rows: list[dict]) -> np.ndarray:
    """Покоординатная медиана невязок (устойчиво к выбросам)."""
    if not rows:
        return np.zeros(3)
    return np.median(np.stack([r["res"] for r in rows]), axis=0)


def mae(rows: list[dict], offset: np.ndarray | None) -> float | None:
    """MAE после вычитания сдвига (None, если точек нет)."""
    if not rows:
        return None
    errs = [float(np.linalg.norm(r["res"] - (offset if offset is not None else 0.0)))
            for r in rows]
    return round(float(np.mean(errs)), 3)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="3D-сдвиг якоря по наборам видов")
    ap.add_argument("--root", default="huniashahatona/zadania")
    ap.add_argument("--repairs", default=None)
    ap.add_argument("--detections", default="out/truck/detections.csv")
    ap.add_argument("--fit-start", type=int, default=0)
    ap.add_argument("--fit-end", type=int, default=1000)
    ap.add_argument("--test-start", type=int, default=600)
    ap.add_argument("--test-end", type=int, default=1000)
    ap.add_argument("--min-observations", type=int, default=20)
    ap.add_argument("--association-mode", default="single_target",
                    choices=["single_target", "epipolar"])
    ap.add_argument("--max-reprojection-px", type=float, default=30.0)
    ap.add_argument("--out", default="datasets/truck/anchor_offset_views.json")
    args = ap.parse_args(argv)

    fusion = FusionConfig(association_mode=args.association_mode,
                          max_reprojection_px=args.max_reprojection_px)
    ds = TruckDataset.discover(args.root, repairs_path=args.repairs)
    dets = load_detections(args.detections)
    if not dets:
        print("нет детекций", file=sys.stderr)
        return 2
    frames = [f for f in sorted(dets)
              if args.fit_start <= f <= args.fit_end or args.test_start <= f <= args.test_end]
    print(f"детекции: кадры {frames[0]}..{frames[-1]} ({len(frames)}), "
          f"фит {args.fit_start}..{args.fit_end}, "
          f"проверка по времени {args.test_start}..{args.test_end}")

    rows = residuals(ds, dets, frames, fusion)
    by_key: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_key[r["views"]].append(r)
    print(f"MEASURED-точек: {len(rows)} в {len(by_key)} наборах видов")

    offsets: dict[str, list[float]] = {}
    per_key: dict[str, dict] = {}
    print(f"\n{'views':18} {'n':>4} {'MAE без':>8} {'MAE in-sample':>13} "
          f"{'parity→':>8} {'time→':>7}")
    for key, items in sorted(by_key.items(), key=lambda kv: -len(kv[1])):
        items = sorted(items, key=lambda r: r["frame"])
        bias = fit(items)
        # ``bias`` — медиана (estimate - gt); в конвейер подаётся обратный
        # вектор: сдвиг прибавляется к оценке, чтобы перейти к pivot.
        correction = -bias
        n = len(items)
        in_sample = mae(items, bias)

        # parity: фит на чётных кадрах -> проверка на нечётных, и наоборот
        even = [r for r in items if r["frame"] % 2 == 0]
        odd = [r for r in items if r["frame"] % 2 == 1]
        p1 = mae(odd, fit(even)) if even and odd else None
        p2 = mae(even, fit(odd)) if even and odd else None
        parity = None if p1 is None else round((p1 + p2) / 2.0, 3)

        # time: фит на fit-диапазоне -> проверка на test-диапазоне
        fit_rows = [r for r in items
                    if args.fit_start <= r["frame"] <= args.fit_end]
        test_rows = [r for r in items
                     if args.test_start <= r["frame"] <= args.test_end]
        time_cv = (mae(test_rows, fit(fit_rows))
                   if fit_rows and test_rows else None)

        entry = {
            "n": n,
            "offset_m": [round(float(v), 4) for v in correction],
            "mae_before_m": mae(items, None),
            "mae_in_sample_m": in_sample,
            "cv_parity_mae_m": parity,
            "cv_time_mae_m": time_cv,
            "cv_time_before_m": mae(test_rows, None) if test_rows else None,
            "cv_time_n": len(test_rows),
        }
        per_key[key] = entry
        if n >= args.min_observations:
            offsets[key] = entry["offset_m"]
        print(f"{key:18} {n:4d} {entry['mae_before_m'] or 0:8.2f} "
              f"{in_sample or 0:13.2f} {parity if parity is not None else float('nan'):8.2f} "
              f"{time_cv if time_cv is not None else float('nan'):7.2f}")

    doc = {
        "method": ("median(estimate - gt) покоординатно по набору видов, знак "
                   "инвертирован: offset_m прибавляется к оценке (переход к pivot); "
                   "канонический ключ — отсортированные имена камер"),
        "offsets_m": offsets,
        "min_observations": args.min_observations,
        "fit_frames": [args.fit_start, args.fit_end],
        "test_frames": [args.test_start, args.test_end],
        "detections_csv": str(args.detections).replace("\\", "/"),
        "per_views": per_key,
        "note": ("mae_in_sample_m — на тех же кадрах, что и фит (не оценка "
                 "точности). Честная проверка переноса — cv_parity_mae_m и "
                 "cv_time_mae_m; если они не лучше mae_before_m, сдвиг для "
                 "этого набора видов не применять."),
    }
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nзаписано: {p} (наборов видов: {len(offsets)} из {len(by_key)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
