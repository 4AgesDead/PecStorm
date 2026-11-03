"""Восстановление и анализ траекторий во времени."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .types import Track, TrackPoint


@dataclass
class TrajectoryFeatures:
    """Скалярные характеристики одной траектории."""

    track_id: int
    n_points: int
    duration_s: float
    path_length: float
    displacement: float
    avg_speed: float
    max_speed: float
    accel_rms: float
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    center: np.ndarray
    mean_reproj_px: float
    mean_views: float


def _as_points(track: Track) -> list[TrackPoint]:
    return sorted(track.points, key=lambda p: p.timestamp)


def trajectory_features(track: Track) -> TrajectoryFeatures:
    """Метрики траектории: длина пути, скорость, ускорение, габариты, качество."""
    pts = _as_points(track)
    if not pts:
        raise ValueError('трек пуст')
    X = np.array([p.position for p in pts], dtype=np.float64)
    T = np.array([p.timestamp for p in pts], dtype=np.float64)
    dt = np.diff(T)
    step = np.linalg.norm(np.diff(X, axis=0), axis=1)
    mask = dt > 1e-9
    speed = np.zeros_like(step)
    speed[mask] = step[mask] / dt[mask]
    dv = np.diff(speed)
    accel_rms = float(np.sqrt(np.mean(dv ** 2))) if dv.size else 0.0
    reproj = [p.mean_reproj_px for p in pts if p.mean_reproj_px >= 0.0]
    views = [len(p.views) for p in pts]
    return TrajectoryFeatures(
        track_id=track.track_id,
        n_points=len(pts),
        duration_s=float(T[-1] - T[0]),
        path_length=float(step.sum()),
        displacement=float(np.linalg.norm(X[-1] - X[0])),
        avg_speed=float(speed.mean()) if speed.size else 0.0,
        max_speed=float(speed.max()) if speed.size else 0.0,
        accel_rms=accel_rms,
        bbox_min=X.min(axis=0),
        bbox_max=X.max(axis=0),
        center=X.mean(axis=0),
        mean_reproj_px=float(np.mean(reproj)) if reproj else -1.0,
        mean_views=float(np.mean(views)) if views else 0.0,
    )


def smooth_positions(track: Track, window: int = 5) -> np.ndarray:
    """Скользящее среднее по позициям (сглаживание «постфактум»)."""
    pts = _as_points(track)
    X = np.array([p.position for p in pts], dtype=np.float64)
    if window <= 1 or X.shape[0] < window:
        return X
    kernel = np.ones(window) / window
    return np.stack([np.convolve(X[:, d], kernel, mode="valid")
                     for d in range(X.shape[1])], axis=1)


def resample(track: Track, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Равномерная передискретизация траектории с шагом ``dt`` (линейная интерполяция)."""
    pts = _as_points(track)
    X = np.array([p.position for p in pts], dtype=np.float64)
    T = np.array([p.timestamp for p in pts], dtype=np.float64)
    if X.shape[0] < 2 or dt <= 0:
        return X, T
    grid = np.arange(T[0], T[-1] + 1e-9, dt)
    out = np.stack([np.interp(grid, T, X[:, d]) for d in range(3)], axis=1)
    return out, grid


@dataclass
class MatchResult:
    """Соответствие «трек ↔ эталонная траектория» и ошибки сопоставления."""

    pairs: list[tuple[int, str]] = field(default_factory=list)
    mean_error_mm: dict[int, float] = field(default_factory=dict)
    unmatched_tracks: list[int] = field(default_factory=list)
    unmatched_truth: list[str] = field(default_factory=list)


def match_tracks_to_truth(tracks: list[Track],
                          truth: dict[str, np.ndarray],
                          max_mean_error: float = 200.0,
                          fps: float | None = None) -> MatchResult:
    """Сопоставление треков с эталоном по средней евклидовой ошибке.

    ``truth`` — словарь «метка -> (N, 3)», строки упорядочены по номеру кадра.
    Если ``fps is None``, таймстемп точки — номер кадра (нумерование симулятора);
    если ``fps`` задан, таймстемпы в секундах и индекс кадра равен
    ``round(timestamp * fps)``. Соответствие решается венгерским алгоритмом по
    средней ошибке на общих кадрах.
    """
    from scipy.optimize import linear_sum_assignment

    def frame_index(p: TrackPoint) -> int:
        return int(round(p.timestamp * fps)) if fps else int(p.timestamp)

    tr = [t for t in tracks if t.points]
    labels = list(truth)
    if not tr or not labels:
        return MatchResult(unmatched_tracks=[t.track_id for t in tr],
                           unmatched_truth=labels)

    cost = np.full((len(tr), len(labels)), np.inf)
    cache: dict[tuple[int, str], float] = {}
    for i, t in enumerate(tr):
        for j, label in enumerate(labels):
            arr = truth[label]
            diffs = [float(np.linalg.norm(np.asarray(p.position) - arr[frame_index(p)]))
                     for p in t.points
                     if 0 <= frame_index(p) < arr.shape[0]]
            if len(diffs) < 2:
                continue
            mean = float(np.mean(diffs))
            cache[(i, label)] = mean
            cost[i, j] = mean

    if np.isinf(cost).all():
        return MatchResult(unmatched_tracks=[t.track_id for t in tr],
                           unmatched_truth=labels)

    filler = (np.nanmin(cost[np.isfinite(cost)]) + 1.0) if np.isfinite(cost).any() else 1e9
    finite = np.where(np.isfinite(cost), cost, filler)
    r, c = linear_sum_assignment(finite)

    res = MatchResult()
    used_tracks, used_labels = set(), set()
    for i, j in zip(r, c):
        if not np.isfinite(cost[i, j]) or cost[i, j] > max_mean_error:
            continue
        track_id = tr[i].track_id
        res.pairs.append((track_id, labels[j]))
        res.mean_error_mm[track_id] = cache[(i, labels[j])]
        used_tracks.add(i)
        used_labels.add(j)
    res.unmatched_tracks = [t.track_id for i, t in enumerate(tr) if i not in used_tracks]
    res.unmatched_truth = [l for j, l in enumerate(labels) if j not in used_labels]
    return res
