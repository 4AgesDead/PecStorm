"""Реконструкция 3D по 2D-детекциям датасета «Пеленга» -> CSV/JSON оценок.

Запуск:
    python apps/reconstruct_truck.py --detections out/truck/detections.csv
        --anchor-offset-views datasets/truck/anchor_offset_views.json
        --out-csv out/truck/estimates.csv --out-json out/truck/estimates.json

Каждый кадр:
* позы камер берутся из camera CSV **на этот кадр** (cam5 вращается) —
  ``CameraStateProvider.calibrations_at(nearest=True)``;
* 2D-детекции сопоставляются по ``association_mode`` (по умолчанию
  ``single_target``: в кадре один грузовик, якорь bbox-центр не равен pivot);
* триангуляция + LM-уточнение, затем трекинг Калмана;
* статус точки: ``MEASURED`` (>=2 вида), ``PREDICTED_ONLY`` (прогноз),
  ``MONOCULAR``/``UNOBSERVABLE`` — см. docs/TRUCK_DATASET.md.

Сдвиг якоря ``anchor_offset_m`` — постоянная поправка из JSON, оценённая на
обучающих кадрах (``tools/anchor_fit_yolo.py``). Если файла нет, сдвиг = 0 и
об этом печатается предупреждение: метрическая ошибка тогда систематическая.
Дополнительно можно подать 2D-сдвиг по камерам ``anchor_px_offset``
(``tools/anchor_fit_px.py``): он вычитается из центров детекций до
триангуляции и компенсирует разное смещение «bbox ↔ pivot» по видам.
Основной калибруемый вариант — ``anchor_offset_views``
(``tools/anchor_fit_views.py``): 3D-сдвиг на набор видов; он проверен
разбиением чётные/нечётные кадры и по времени (см. docs/RESULTS.md).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from multiview3d.fusion import FusionConfig
from multiview3d.truck_dataset import DEFAULT_REPAIRS, DEFAULT_ROOT, TruckDataset
from multiview3d.truck_pipeline import (
    estimate_points,
    export_estimates,
    load_anchor_offset,
    load_anchor_offset_views,
    load_anchor_px_offset,
    load_detections,
    run_reconstruction,
    write_estimates_json,
)
from multiview3d.types import MEASURED, PREDICTED_ONLY


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Реконструкция 3D по датасету «Пеленга»")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--repairs", default=str(DEFAULT_REPAIRS))
    ap.add_argument("--detections", required=True)
    ap.add_argument("--anchor-offset", default="datasets/truck/anchor_offset.json")
    ap.add_argument("--anchor-px-offset", default="datasets/truck/anchor_px_offset.json")
    ap.add_argument("--anchor-offset-views", default="datasets/truck/anchor_offset_views.json")
    ap.add_argument("--no-anchor-offset", action="store_true")
    ap.add_argument("--no-anchor-px-offset", action="store_true")
    ap.add_argument("--no-anchor-offset-views", action="store_true")
    ap.add_argument("--association-mode", default="single_target",
                    choices=["single_target", "epipolar"])
    ap.add_argument("--min-views", type=int, default=2)
    ap.add_argument("--max-reprojection-px", type=float, default=50.0,
                    help="порог средней невязки триангуляции; для bbox-якорей "
                         "десятки px — норма (разные точки объекта в видах); "
                         "50 px выбран по experiments/geometry_sweep.py")
    ap.add_argument("--gate", type=float, default=10.0,
                    help="гейт махаланобиса (sigma); 10 подобран по свипу: "
                         "фрагментация треков падает с 486 до ~45 без потери "
                         "точности измерений (docs/RESULTS.md §8e)")
    ap.add_argument("--max-missing", type=int, default=3)
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--out-csv", default="out/truck/estimates.csv")
    ap.add_argument("--out-json", default="out/truck/estimates.json")
    ap.add_argument("--log-every", type=int, default=100)
    args = ap.parse_args(argv)

    ds = TruckDataset.discover(args.root, repairs_path=args.repairs or None)
    detections = load_detections(args.detections)
    if not detections:
        print(f"в {args.detections} нет детекций", file=sys.stderr)
        return 2
    frames = sorted(detections)
    lo, hi = (args.start, args.end)
    frames = [f for f in frames
              if (lo is None or f >= lo) and (hi is None or f <= hi)]

    offset = None
    if not args.no_anchor_offset:
        offset = load_anchor_offset(args.anchor_offset)
        if offset is not None:
            print(f"Сдвиг якоря: dx={offset[0]:+.3f} dy={offset[1]:+.3f} "
                  f"dz={offset[2]:+.3f} м (из {args.anchor_offset})")

    px_offset = None
    if not args.no_anchor_px_offset:
        px_offset = load_anchor_px_offset(args.anchor_px_offset)
        if px_offset is not None:
            pretty = ", ".join(f"{k}: du={v[0]:+.1f} dv={v[1]:+.1f}"
                               for k, v in sorted(px_offset.items()))
            print(f"2D-сдвиг якоря (вычитается до триангуляции): {pretty} px "
                  f"(из {args.anchor_px_offset})")

    views_offset = None
    if not args.no_anchor_offset_views:
        views_offset = load_anchor_offset_views(args.anchor_offset_views)
        if views_offset is None:
            print(f"ВНИМАНИЕ: {args.anchor_offset_views} не найден — сдвиг по "
                  f"наборам видов = 0 (см. tools/anchor_fit_views.py)")
        else:
            pretty = ", ".join(
                f"{k}: ({v[0]:+.2f}, {v[1]:+.2f}, {v[2]:+.2f})"
                for k, v in sorted(views_offset.items()))
            print(f"Сдвиг якоря по наборам видов (м): {pretty} "
                  f"(из {args.anchor_offset_views})")

    fusion = FusionConfig(
        association_mode=args.association_mode,
        min_views=args.min_views,
        max_reprojection_px=args.max_reprojection_px,
        gate_mahalanobis=args.gate,
        max_missing=args.max_missing,
    )
    result = run_reconstruction(ds, detections, fusion=fusion, anchor_offset_m=offset,
                                anchor_px_offset=px_offset,
                                anchor_offset_views=views_offset,
                                frames=frames, log_every=args.log_every)
    pts = estimate_points(result)
    n_csv = export_estimates(result, args.out_csv)
    doc = write_estimates_json(result, args.out_json, meta={
        "dataset": "pelenga_truck",
        "fps": ds.fps,
        "frame_range": [frames[0], frames[-1]],
        "detections_csv": str(args.detections).replace("\\", "/"),
        "anchor_offset_m": offset,
        "anchor_px_offset": px_offset,
        "anchor_offset_views": views_offset,
        "fusion": {
            "association_mode": fusion.association_mode,
            "min_views": fusion.min_views,
            "max_reprojection_px": fusion.max_reprojection_px,
            "gate_mahalanobis": fusion.gate_mahalanobis,
            "max_missing": fusion.max_missing,
        },
    })

    n_meas = sum(1 for p in pts if p.status == MEASURED)
    n_pred = sum(1 for p in pts if p.status == PREDICTED_ONLY)
    cov = result.pipeline.coverage
    print(f"Кадров: {len(frames)}; точек в треках: {len(pts)} "
          f"(MEASURED {n_meas}, PREDICTED_ONLY {n_pred})")
    print(f"Покрытие кадров: измерение {cov['measured']}, только прогноз "
          f"{cov['predicted_only']}, без оценки {cov['none']}")
    print(f"Гистограмма числа видов: {cov['views_histogram']}")
    print(f"Время реконструкции: {result.seconds:.1f} с "
          f"({1000.0 * result.seconds_per_frame:.2f} мс/кадр)")
    print(f"Оценки: {args.out_csv} ({n_csv} строк), {args.out_json} "
          f"({len(doc['estimates'])} записей)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
