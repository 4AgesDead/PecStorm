"""Метрики качества: ошибка 3D по ground truth, coverage, ID-switch, полнота.

Единицы: позиции треков и эталона — в мировых единицах калибровки
(``CameraCalibration.t``); ошибка пересчитывается в миллиметры множителем
``unit`` (``unit=1.0`` — метры, следовательно мм = error * unit * 1000).
Точная формула: ``mm = ||pos - truth|| * unit * 1000``.

Ключевое требование §7.6 брифа: **coverage отдельно от error**. Оценка на
реальных данных (``evaluate_truck``) считает MAE/RMSE/P95/P99 только по
измерениям (``status == MEASURED``), а долю кадров с прогнозом
(``PREDICTED_ONLY``), однокамерных и без оценки выводит отдельно — иначе
«точность» на лёгких интервалах выглядит лучше, чем есть.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from .trajectory import MatchResult, match_tracks_to_truth
from .types import MEASURED, PREDICTED_ONLY, Track, TrackPoint


@dataclass
class ErrorStats:
    """Статистика ошибки в миллиметрах: MAE/RMSE/медиана/P95/P99/max + по осям."""

    n_points: int = 0
    mean_mm: float = float("nan")
    rmse_mm: float = float("nan")
    median_mm: float = float("nan")
    max_mm: float = float("nan")
    p95_mm: float = float("nan")
    p99_mm: float = float("nan")
    per_axis_mae_mm: list[float] | None = None
    per_axis_rmse_mm: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_points": self.n_points,
            "mean_mm": _round(self.mean_mm),
            "mae_mm": _round(self.mean_mm),
            "rmse_mm": _round(self.rmse_mm),
            "median_mm": _round(self.median_mm),
            "p95_mm": _round(self.p95_mm),
            "p99_mm": _round(self.p99_mm),
            "max_mm": _round(self.max_mm),
            "per_axis_mae_mm": (None if self.per_axis_mae_mm is None
                                else [_round(v) for v in self.per_axis_mae_mm]),
            "per_axis_rmse_mm": (None if self.per_axis_rmse_mm is None
                                 else [_round(v) for v in self.per_axis_rmse_mm]),
        }


def _round(v: float) -> float | None:
    return float(round(v, 3)) if np.isfinite(v) else None


def _stats(errors_mm: np.ndarray, deltas_mm: np.ndarray | None = None) -> ErrorStats:
    """Статистика по массиву ошибок; ``deltas_mm`` — (N,3) покомпонентные невязки."""
    if errors_mm.size == 0:
        return ErrorStats()
    st = ErrorStats(
        n_points=int(errors_mm.size),
        mean_mm=float(np.mean(errors_mm)),
        rmse_mm=float(np.sqrt(np.mean(errors_mm ** 2))),
        median_mm=float(np.median(errors_mm)),
        max_mm=float(np.max(errors_mm)),
        p95_mm=float(np.percentile(errors_mm, 95)),
        p99_mm=float(np.percentile(errors_mm, 99)),
    )
    if deltas_mm is not None and deltas_mm.size:
        st.per_axis_mae_mm = [float(v) for v in np.mean(np.abs(deltas_mm), axis=0)]
        st.per_axis_rmse_mm = [float(v) for v in np.sqrt(np.mean(deltas_mm ** 2, axis=0))]
    return st


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
        k = p.frame if p.frame is not None else int(round(p.timestamp * fps))
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
            k = p.frame if p.frame is not None else int(round(p.timestamp * fps))
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


# --------------------------------------------------------------------------
# Оценка на реальных данных (датасет «Пеленга»): coverage отдельно от ошибки
# --------------------------------------------------------------------------


@dataclass
class TruckEvaluation:
    """Итог оценки одного объекта: ошибки по измерениям + покрытие по статусам."""

    measured: ErrorStats = field(default_factory=ErrorStats)
    predicted: ErrorStats = field(default_factory=ErrorStats)
    frames_total: int = 0
    frames_with_gt: int = 0
    frames_measured: int = 0
    frames_predicted_only: int = 0
    frames_unobservable: int = 0
    frames_monocular_only: int = 0
    per_camera_subset: dict[str, ErrorStats] = field(default_factory=dict)
    per_frame: list[dict[str, Any]] = field(default_factory=list)
    gt_invalid_frames: list[int] = field(default_factory=list)

    def coverage(self) -> dict[str, float]:
        total = max(self.frames_total, 1)
        return {
            "frames_total": float(self.frames_total),
            "frames_with_gt": float(self.frames_with_gt),
            "measured": self.frames_measured / total,
            "progress_predicted_only": self.frames_predicted_only / total,
            "unobservable": self.frames_unobservable / total,
            "monocular_only": self.frames_monocular_only / total,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "measured": self.measured.to_dict(),
            "predicted": self.predicted.to_dict(),
            "coverage": {k: _round(v) for k, v in self.coverage().items()},
            "coverage_counts": {
                "frames_total": self.frames_total,
                "frames_with_gt": self.frames_with_gt,
                "frames_measured": self.frames_measured,
                "frames_predicted_only": self.frames_predicted_only,
                "frames_unobservable": self.frames_unobservable,
                "frames_monocular_only": self.frames_monocular_only,
            },
            "per_camera_subset": {k: v.to_dict() for k, v in self.per_camera_subset.items()},
            "gt_invalid_frames": self.gt_invalid_frames,
        }


def _frame_of(point: TrackPoint, fps: float) -> int:
    return point.frame if point.frame is not None else int(round(point.timestamp * fps))


def evaluate_truck(points: Iterable[TrackPoint],
                   gt_by_frame: dict[int, np.ndarray],
                   *,
                   frames: Sequence[int] | None = None,
                   fps: float = 30.0,
                   unit: float = 1.0,
                   gt_invalid_frames: Sequence[int] | None = None) -> TruckEvaluation:
    """Оценка оценок 3D против GT на реальных данных.

    * ошибка считается **только** по точкам ``MEASURED``, у которых есть GT;
    * ``PREDICTED_ONLY`` даёт отдельную статистику (это прогноз, не измерение);
    * покрытие — доли кадров с измерением / только прогнозом / без оценки;
    * ``frames`` — полный список кадров прогона (по умолчанию — объединение
      кадров оценок и ключей GT).

    Точки разных треков объединяются (в наборе один объект); при нескольких
    треках на одном кадре берётся точка с наибольшей ``confidence``.
    """
    pts = list(points)
    by_frame: dict[int, TrackPoint] = {}
    for p in pts:
        f = _frame_of(p, fps)
        cur = by_frame.get(f)
        if cur is None or _better(p, cur):
            by_frame[f] = p

    if frames is None:
        all_frames = sorted(set(by_frame) | set(gt_by_frame))
        frames = all_frames
    frames = [int(f) for f in frames]

    rep = TruckEvaluation()
    rep.frames_total = len(frames)
    rep.gt_invalid_frames = [int(f) for f in (gt_invalid_frames or []) if int(f) in set(frames)]

    errs_measured: list[float] = []
    errs_predicted: list[float] = []
    deltas_measured: list[np.ndarray] = []
    deltas_predicted: list[np.ndarray] = []
    subset_err: dict[str, list[float]] = {}

    for f in frames:
        p = by_frame.get(f)
        gt = gt_by_frame.get(f)
        row: dict[str, Any] = {"frame": f, "status": None, "method": None,
                               "views": [], "confidence": None}
        if gt is not None:
            rep.frames_with_gt += 1
        if p is None:
            rep.frames_unobservable += 1
            row["status"] = "UNOBSERVABLE"
            row["method"] = "none"
            rep.per_frame.append(row)
            continue
        row.update({
            "timestamp": float(p.timestamp),
            "status": p.status,
            "method": p.method,
            "views": list(p.views),
            "confidence": (float(p.confidence) if np.isfinite(p.confidence) else None),
            "x": float(p.position[0]), "y": float(p.position[1]), "z": float(p.position[2]),
        })
        if p.status == MEASURED:
            rep.frames_measured += 1
            if len(p.views) == 1:
                rep.frames_monocular_only += 1
        elif p.status == PREDICTED_ONLY:
            rep.frames_predicted_only += 1
        if gt is not None:
            d = (np.asarray(p.position, dtype=float) - np.asarray(gt, dtype=float)) * unit * 1000.0
            err = float(np.linalg.norm(d))
            row["error_mm"] = err
            row["gt_x"], row["gt_y"], row["gt_z"] = [float(v) for v in gt]
            if p.status == MEASURED:
                errs_measured.append(err)
                deltas_measured.append(d)
                key = "|".join(p.views) if p.views else "none"
                subset_err.setdefault(key, []).append(err)
            elif p.status == PREDICTED_ONLY:
                errs_predicted.append(err)
                deltas_predicted.append(d)
        rep.per_frame.append(row)

    rep.measured = _stats(np.asarray(errs_measured),
                          np.asarray(deltas_measured) if deltas_measured else None)
    rep.predicted = _stats(np.asarray(errs_predicted),
                           np.asarray(deltas_predicted) if deltas_predicted else None)
    rep.per_camera_subset = {k: _stats(np.asarray(v)) for k, v in sorted(subset_err.items())}
    return rep


def _better(candidate: TrackPoint, current: TrackPoint) -> bool:
    """Лучшая точка кадра: измерение важнее прогноза, затем выше confidence."""
    if candidate.status != current.status:
        return candidate.status == MEASURED
    c = candidate.confidence if np.isfinite(candidate.confidence) else -1.0
    k = current.confidence if np.isfinite(current.confidence) else -1.0
    return c > k


def per_frame_csv_rows(report: TruckEvaluation) -> list[dict[str, Any]]:
    """Строки для CSV отчёта (по кадрам): ошибка, статус, метод, виды."""
    out = []
    for row in report.per_frame:
        out.append({
            "frame": row["frame"],
            "status": row.get("status"),
            "method": row.get("method") or "",
            "views": "|".join(row.get("views") or []),
            "x": row.get("x", ""),
            "y": row.get("y", ""),
            "z": row.get("z", ""),
            "gt_x": row.get("gt_x", ""),
            "gt_y": row.get("gt_y", ""),
            "gt_z": row.get("gt_z", ""),
            "error_mm": row.get("error_mm", ""),
            "confidence": row.get("confidence", ""),
        })
    return out
