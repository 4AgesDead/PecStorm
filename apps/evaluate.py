"""Оценка треков против ground truth по CSV-файлам.

Запуск:
    python -m apps.evaluate --tracks out/tracks.csv --truth out/truth.csv [--fps 25] [--unit 1.0]

Форматы — те, что пишет ``multiview3d.visualize`` (``tracks_to_csv``,
``truth_to_csv``). Частота кадров определяется по таймкодам, если не задана.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from multiview3d.metrics import EvaluationReport, evaluate
from multiview3d.visualize import tracks_from_csv, truth_from_csv


def infer_fps(times: list[float], fps_arg: float | None) -> float:
    if fps_arg:
        return float(fps_arg)
    if len(times) < 2:
        raise ValueError("не могу определить fps: меньше двух таймкодов")
    dt = np.diff(np.asarray(sorted(times), dtype=float))
    dt = dt[dt > 1e-9]
    if dt.size == 0:
        raise ValueError("таймкоды совпадают — fps не определён")
    return float(1.0 / np.median(dt))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Оценка треков по ground truth (CSV)")
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--unit", type=float, default=1.0,
                    help="Множитель перевода мировых единиц в метры (1.0 = метры)")
    args = ap.parse_args(argv)

    tracks = tracks_from_csv(args.tracks)
    truth, times = truth_from_csv(args.truth)
    fps = infer_fps(times, args.fps)
    labels = list(truth)
    truth_points = [truth[label] for label in labels]

    rep: EvaluationReport = evaluate(tracks, truth_points, labels, fps=fps, unit=args.unit)
    print(f"Частота кадров: {fps:.2f} fps; треков: {len(tracks)}; эталонных объектов: {len(labels)}")
    print(f"Сопоставлено: {rep.pairs}")
    if rep.unmatched_tracks:
        print(f"Не сопоставленные треки: {rep.unmatched_tracks}")
    if rep.unmatched_truth:
        print(f"Не сопоставленные эталоны: {rep.unmatched_truth}")
    print(f"Ошибка 3D (мм): среднее={rep.overall.mean_mm}, max={rep.overall.max_mm}, "
          f"p95={rep.overall.p95_mm}, точек={rep.overall.n_points}")
    for tid, mean in rep.per_track_mean_mm.items():
        print(f"  трек {tid}: среднее {mean:.1f} мм")
    print(f"Полнота: {rep.completeness}")
    print(f"ID-switch: {rep.id_switches}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
