"""Метрики качества: ошибка 3D по ground truth, ID-switch, полнота треков.

Единцы: позиции треков и эталона — в мировых единицах калибровки
(``CameraCalibration.t``); ошибка пересчитывается в миллиметры множителем
``unit`` (``unit=1.0`` — метры, следовательно мм = error * unit * 1000).
Точная формула: ``mm = ||pos - truth|| * unit * 1000``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .trajectory import MatchResult, match_tracks_to_truth
from .types import Track


@dataclass
class ErrorStats:
    """Сводная статистика ошибки в миллиметрах по сопоставленным точкам."""

    n_points: int = 0
    mean_mm: float = float("nan")
    max_mm: float = float("nan")
    p95_mm: float = float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_points": self.n_points,
            "mean_mm": _round(self.mean_mm),
            "max_mm": _round(self.max_mm),
            "p95_mm": _round(self.p95_mm),
        }


def _round(v: float) -> float:
    return float(round(v, 3)) if np.isfinite(v) else None


def _stats(errors_mm: np.ndarray) -> ErrorStats:
    if errors_mm.size == 0:
        return ErrorStats()
    return ErrorStats(
        n_points=int(errors_mm.size),
        mean_mm=float(np.mean(errors_mm)),
        max_mm=float(np.max(errors_mm)),
        p95_mm=float(np.percentile(errors_mm, 95)),
    )


def build_truth_dict(truth_points: list[list[np.ndarray]],
                     labels: list[str]) -> dict[str, np.ndarray]:
    """Эталон «метка -> (N, 3)»: ``truth_points[k][i]`` — позиция объекта k в кадре i."""
    return {label: np.asarray(points, dtype=float).reshape(-1, 3)
            for label, points in zip(labels, truth_points)}


def point_errors_mm(track: Track, truth_arr: np.ndarray, fps: float,
                    unit: float = 1.0) -> np.ndarray:
    """Ошибки точек трека (мм) против эталона ``(N, 3)``; индекс кадра = round(t*fps)."""
    errs = []
    for p in track.points:
        k = int(round(p.timestamp * fps))
        if 0 <= k < truth_arr.shape[0]:
            errs.append(float(np.linalg.norm(np.asarray(p.position) - truth_arr[k]))
                        * unit * 1000.0)
    return np.asarray(errs, dtype=float)


def count_id_switches(tracks: list[Track], truth_dict: dict[str, np.ndarray],
                      fps: float, unit: float = 1.0,
                      switch_radius_mm: float = 150.0) -> int:
    """Число смен метки ground truth у одного трека во времени.

    Каждая точка трека сопоставляется ближайшему эталонному объекту, если
    расстояние меньше ``switch_radius_mm``; переход между разными метками —
    ID-switch. Трек, не сопоставленный ни с кем, не порождает переключений.
    """
    labels = list(truth_dict)
    if not labels:
        return 0
    n_frames = next(iter(truth_dict.values())).shape[0]
    total = 0
    for track in tracks:
        current: str | None = None
        for p in track.points:
            k = int(round(p.timestamp * fps))
            if not (0 <= k < n_frames):
                continue
            best, best_d = None, np.inf
            for label in labels:
                d = float(np.linalg.norm(np.asarray(p.position) - truth_dict[label][k]))
                if d < best_d:
                    best, best_d = label, d
            if best is None or best_d * unit * 1000.0 > switch_radius_mm:
                continue
            if current is not None and best != current:
                total += 1
            current = best
    return total


@dataclass
class EvaluationReport:
    """Итог оценки: соответствие треков, ошибки, полнота, ID-switch, темп."""

    pairs: list[tuple[int, str]] = field(default_factory=list)
    per_track_mean_mm: dict[int, float] = field(default_factory=dict)
    overall: ErrorStats = field(default_factory=ErrorStats)
    completeness: dict[str, float] = field(default_factory=dict)
    id_switches: int = 0
    unmatched_tracks: list[int] = field(default_factory=list)
    unmatched_truth: list[str] = field(default_factory=list)
    processing_fps: float = float("nan")
    step_ms_mean: float = float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "pairs": [[tid, label] for tid, label in self.pairs],
            "per_track_mean_mm": {str(k): _round(v) for k, v in self.per_track_mean_mm.items()},
            "overall": self.overall.to_dict(),
            "completeness": {k: _round(v) for k, v in self.completeness.items()},
            "id_switches": self.id_switches,
            "unmatched_tracks": self.unmatched_tracks,
            "unmatched_truth": self.unmatched_truth,
            "processing_fps": _round(self.processing_fps),
            "step_ms_mean": _round(self.step_ms_mean),
        }


def evaluate(tracks: list[Track],
             truth_points: list[list[np.ndarray]],
             labels: list[str],
             fps: float,
             unit: float = 1.0,
             max_mean_error: float = 200.0,
             pipeline_stats: dict[str, Any] | None = None) -> EvaluationReport:
    """Оценка треков против ground truth синтетики.

    ``truth_points[k]`` — эталонные позиции объекта ``labels[k]`` по всем кадрам.
    Сопоставление «трек <-> объект» — венгерское по средней ошибке
    (``trajectory.match_tracks_to_truth`` с таймкодами в секундах).
    """
    truth_dict = build_truth_dict(truth_points, labels)
    match: MatchResult = match_tracks_to_truth(tracks, truth_dict,
                                               max_mean_error=max_mean_error, fps=fps)
    report = EvaluationReport(
        pairs=match.pairs,
        unmatched_tracks=match.unmatched_tracks,
        unmatched_truth=match.unmatched_truth,
    )

    by_id = {t.track_id: t for t in tracks}
    all_errors: list[np.ndarray] = []
    matched_count: dict[str, int] = {}
    for tid, label in match.pairs:
        errs = point_errors_mm(by_id[tid], truth_dict[label], fps, unit)
        report.per_track_mean_mm[tid] = float(np.mean(errs)) if errs.size else float("nan")
        all_errors.append(errs)
        matched_count[label] = matched_count.get(label, 0) + int(errs.size)

    report.overall = _stats(np.concatenate(all_errors) if all_errors else np.empty(0))
    n_frames = max((len(p) for p in truth_points), default=0)
    report.completeness = {label: (matched_count.get(label, 0) / n_frames)
                           if n_frames else float("nan") for label in labels}
    report.id_switches = count_id_switches(tracks, truth_dict, fps, unit)
    if pipeline_stats:
        report.processing_fps = float(pipeline_stats.get("processing_fps", float("nan")))
        report.step_ms_mean = float(pipeline_stats.get("step_ms_mean", float("nan")))
    return report
