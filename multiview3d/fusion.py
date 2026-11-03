"""Сопоставление видов между камерами, триангуляция и трекинг 3D-объектов.

Конвейер одного синхронного кадра:
1. Симметричные эпиполярные стоимости между всеми парами камер
   (``calibration.correspondence_epipolar_errors``).
2. Венгерское сопоставление по парам (``scipy.optimize.linear_sum_assignment``)
   и склейка пар в группы: на каждую камеру — не более одной детекции.
3. DLT-триангуляция лучей (``camera.triangulate_rays``) + уточнение
   Левенбергом–Марквардом по репроекции (``camera.refine_point``).
4. Ассоциация 3D-кандидатов с активными треками: предсказание фильтром
   Калмана (CV-модель, 6D) и гейт по махаланобисову расстоянию.

Единицы мировые — те же, что у ``CameraCalibration.t`` (обычно метры).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .camera import (
    back_project_ray,
    reprojection_error,
    refine_point,
    triangulate_rays,
)
from .calibration import correspondence_epipolar_errors
from .kalman import KalmanFilter
from .types import CameraCalibration, Detection2D, Observation3D, TrackPoint


@dataclass
class FusionConfig:
    """Пороговые параметры сопоставления, триангуляции и трекинга."""

    epipolar_threshold_px: float = 8.0
    """Максимальная симметричная эпиполярная стоимость (сумма двух направлений, px)."""
    max_reprojection_px: float = 4.0
    """Максимальная средняя ошибка репроекции триангулированной точки (px)."""
    min_views: int = 2
    """Минимальное число видов для принятия кандидата."""
    gate_mahalanobis: float = 3.0
    """Гейт ассоциации: sqrt(махаланобисова расстояния) не больше порога."""
    max_missing: int = 3
    """Число шагов без измерений, после которого трек снимается с активной."""
    position_noise: float = 0.02
    velocity_noise: float = 1.20
    """Дисперсия прироста скорости на 1 с: должно покрывать маневрирование цели
    (иначе CV-фильтр «запирает» скорость, гейт рвётся и трек фрагментируется)."""
    measurement_noise: float = 0.04
    workspace_bounds: tuple[float, float, float, float, float, float] | None = None
    """``(x_min, y_min, z_min, x_max, y_max, z_max)`` или None."""


@dataclass
class Candidate3D:
    """Триангулированная точка с поддерживающими детекциями."""

    position: np.ndarray
    views: tuple[Detection2D, ...]
    mean_reproj_px: float

    def view_names(self) -> list[str]:
        return [v.camera for v in self.views]


@dataclass
class TrackedObject:
    """Активный трек с фильтром Калмана (CV-модель) и историей для RTS."""

    track_id: int
    filter: KalmanFilter
    last_update: float
    missing: int = 0
    observations: list[TrackPoint] = field(default_factory=list)
    last_views: tuple[Detection2D, ...] = ()
    last_reproj_px: float = 0.0
    pending: bool = True
    """True до второго наблюдения: скорость инициализируется по первой паре точек."""
    birth_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    birth_time: float = 0.0
    # История (t, x, P) после update() — для rts_smooth снаружи.
    state_history: list[tuple[float, np.ndarray, np.ndarray]] = field(default_factory=list)

    @property
    def position(self) -> np.ndarray:
        return np.asarray(self.filter.x[:3], dtype=float)

    @property
    def velocity(self) -> np.ndarray:
        return np.asarray(self.filter.x[3:], dtype=float)


@dataclass
class FrameFusionResult:
    """Результат обработки одного синхронизированного кадра."""

    timestamp: float
    candidates: list[Candidate3D]
    observations: list[Observation3D]
    tracks: list[TrackedObject]
    rejected: list[str]


def cam_center(cam: CameraCalibration) -> np.ndarray:
    """Центр проекции камеры: ``C = -R^T t`` (в ``camera.py`` такой функции нет)."""
    return -np.asarray(cam.R, dtype=float).T @ np.asarray(cam.t, dtype=float).reshape(3)


def _hungarian_pairs(cost: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Венгерское сопоставление по матрице стоимостей; пары выше порога отбрасываются."""
    finite = np.isfinite(cost)
    if not finite.any():
        return []
    filler = float(np.min(cost[finite])) + threshold + 1.0
    r, c = linear_sum_assignment(np.where(finite, cost, filler))
    return [(int(i), int(j)) for i, j in zip(r, c) if finite[i, j] and cost[i, j] <= threshold]


def build_correspondences(
    cameras: list[CameraCalibration],
    detections: list[list[Detection2D]],
    config: FusionConfig,
) -> tuple[list[dict[int, int]], list[str]]:
    """Группы соответствий ``{индекс камеры: индекс детекции}`` по парным эпиполярным стоимостям.

    Пары сопоставляются венгерским алгоритмом и склеиваются в транзитивные
    группы; конфликтные склейки (камера уже занята в другой группе) заносятся
    в ``rejected``.
    """
    n = len(cameras)
    groups: list[dict[int, int]] = []
    owner: dict[tuple[int, int], int] = {}  # (camera_idx, det_idx) -> номер группы
    rejected: list[str] = []

    def attach(cam_i: int, det_i: int, group_id: int) -> bool:
        g = groups[group_id]
        if cam_i in g:
            return g[cam_i] == det_i
        g[cam_i] = det_i
        owner[(cam_i, det_i)] = group_id
        return True

    for i, j in combinations(range(n), 2):
        di, dj = detections[i], detections[j]
        if not di or not dj:
            continue
        errors = correspondence_epipolar_errors(
            cameras[i], cameras[j],
            np.array([d.center for d in di]),
            np.array([d.center for d in dj]),
        )
        for a, b in _hungarian_pairs(errors, config.epipolar_threshold_px):
            gi = owner.get((i, a))
            gj = owner.get((j, b))
            if gi is not None and gj is not None:
                if gi != gj:
                    rejected.append(
                        f"conflict: cam{i}#{a} ({gi}) ↔ cam{j}#{b} ({gj})"
                    )
                continue
            gid = gi if gi is not None else gj
            if gid is None:
                groups.append({i: a, j: b})
                gid = len(groups) - 1
                owner[(i, a)] = gid
                owner[(j, b)] = gid
            else:
                if not attach(i, a, gid) or not attach(j, b, gid):
                    rejected.append(f"conflict: cam{i}#{a} ↔ cam{j}#{b}")
    return groups, rejected


def triangulate_group(
    cameras: list[CameraCalibration],
    group: dict[int, int],
    detections: list[list[Detection2D]],
    config: FusionConfig,
    timestamp: float,
) -> Candidate3D | None:
    """Триангуляция одной группы видов; None, если точка не проходит пороги."""
    cam_ids = sorted(group)
    cams = [cameras[ci] for ci in cam_ids]
    pixels = [detections[ci][group[ci]].center for ci in cam_ids]
    views = tuple(detections[ci][group[ci]] for ci in cam_ids)
    if len(cams) < config.min_views:
        return None

    rays = [back_project_ray(cam, px) for cam, px in zip(cams, pixels)]
    origins = [cam_center(cam) for cam in cams]
    X = triangulate_rays(rays, origins)
    if not np.isfinite(X).all():
        return None
    X = refine_point(cams, pixels, X)
    if not np.isfinite(X).all():
        return None

    errs = [reprojection_error(cam, X, px) for cam, px in zip(cams, pixels)]
    if not np.isfinite(errs).all() or max(errs) > config.max_reprojection_px * 2:
        return None
    mean_px = float(np.mean(errs))
    if mean_px > config.max_reprojection_px:
        return None

    if config.workspace_bounds is not None:
        lo = np.asarray(config.workspace_bounds[:3], dtype=float)
        hi = np.asarray(config.workspace_bounds[3:], dtype=float)
        if not (np.all(X >= lo) and np.all(X <= hi)):
            return None
    return Candidate3D(position=X, views=views, mean_reproj_px=mean_px)


def _make_filter(position: np.ndarray, timestamp: float, config: FusionConfig,
                 velocity: np.ndarray | None = None, dt_hint: float | None = None) -> KalmanFilter:
    """CV-фильтр.

    При ``velocity is None`` (первое наблюдение) скорость нулевая, ковариация
    скорости широкая. При скорости из первых двух наблюдений ковариация учитывает
    шум обоих измерений и добавляет перекрёстную связь позиция<->скорость
    (без неё фильтр Калмана по одним позициям почти не учится оценивать скорость).
    """
    pos_var = config.position_noise ** 2
    vel_var = config.velocity_noise ** 2
    v = np.zeros(3) if velocity is None else np.asarray(velocity, dtype=float)
    x0 = np.concatenate((np.asarray(position, dtype=float), v))
    if velocity is None or dt_hint is None or dt_hint <= 1e-9:
        p0 = np.diag([pos_var] * 3 + [4.0 * vel_var] * 3)
    else:
        meas_var = max(config.measurement_noise ** 2, pos_var)
        # Ошибка скорости из двух измерений: 2*meas_var/dt^2; связь Cov(pos, v) = meas_var/dt.
        p_vv = max(2.0 * meas_var / dt_hint ** 2, vel_var)
        p_pv = min(meas_var / dt_hint, (pos_var * p_vv) ** 0.5 * 0.9)
        p0 = np.diag([pos_var] * 3 + [p_vv] * 3)
        for i in range(3):
            p0[i, 3 + i] = p_pv
            p0[3 + i, i] = p_pv
    filt = KalmanFilter(
        x0, p0,
        (config.position_noise ** 2, config.velocity_noise ** 2),
        config.measurement_noise ** 2,
    )
    filt.t = float(timestamp)
    return filt


class MultiViewTracker:
    """Трекинг 3D-объектов: группы видов -> кандидаты -> треки с гейтом Калмана."""

    def __init__(
        self,
        camera_names: list[str],
        cameras: list[CameraCalibration],
        config: FusionConfig | None = None,
    ) -> None:
        if len(camera_names) != len(cameras):
            raise ValueError("camera_names и cameras должны быть одинаковой длины")
        self.camera_names = list(camera_names)
        self.cameras = list(cameras)
        self.config = config or FusionConfig()
        self._tracks: list[TrackedObject] = []
        self._next_id = 1

    def active_tracks(self) -> list[TrackedObject]:
        return list(self._tracks)

    def step(
        self, timestamp: float, detections: list[list[Detection2D]]
    ) -> FrameFusionResult:
        """Обрабатывает синхронный кадр детекций (порядок совпадает с ``cameras``)."""
        groups, rejected = build_correspondences(self.cameras, detections, self.config)
        candidates: list[Candidate3D] = []
        for group in groups:
            cand = triangulate_group(self.cameras, group, detections, self.config, timestamp)
            if cand is None:
                rejected.append(
                    "triangulate rejected: views=" + ",".join(
                        self.camera_names[ci] for ci in sorted(group)
                    )
                )
                continue
            candidates.append(cand)

        observations = [
            Observation3D(
                timestamp=timestamp,
                position=cand.position.copy(),
                views=cand.view_names(),
                mean_reproj_px=cand.mean_reproj_px,
            )
            for cand in candidates
        ]

        # Ассоциация кандидатов с активными треками: гейт по махаланобису.
        for track in self._tracks:
            track.filter.predict(timestamp)
        gate = self.config.gate_mahalanobis
        if self._tracks and candidates:
            cost = np.full((len(self._tracks), len(candidates)), np.inf)
            for ti, track in enumerate(self._tracks):
                for ci, cand in enumerate(candidates):
                    m2 = track.filter.mahalanobis(cand.position)
                    if np.isfinite(m2) and np.sqrt(max(m2, 0.0)) <= gate:
                        cost[ti, ci] = float(np.sqrt(max(m2, 0.0)))
            pairs = _hungarian_pairs(cost, gate)
            matched_t = [p[0] for p in pairs]
            matched_c = [p[1] for p in pairs]
        else:
            matched_t, matched_c = [], []

        used_c: set[int] = set()
        used_t: set[int] = set()
        for ti, ci in zip(matched_t, matched_c):
            track, cand = self._tracks[ti], candidates[ci]
            if track.pending:
                # Второе наблюдение: скорость по первой паре точек — иначе CV-модель
                # стартует с нулевой скоростью и отстаёт от маневрирующего объекта.
                dtv = timestamp - track.birth_time
                vel = (cand.position - track.birth_position) / dtv if dtv > 1e-9 else np.zeros(3)
                track.filter = _make_filter(cand.position, timestamp, self.config,
                                            velocity=vel, dt_hint=dtv)
                track.pending = False
            else:
                track.filter.update(cand.position)
            track.last_update = timestamp
            track.missing = 0
            track.last_views = cand.views
            track.last_reproj_px = cand.mean_reproj_px
            track.observations.append(
                TrackPoint(
                    timestamp=timestamp,
                    position=track.position.copy(),
                    velocity=track.velocity.copy(),
                    views=cand.view_names(),
                    mean_reproj_px=cand.mean_reproj_px,
                )
            )
            track.state_history.append(
                (timestamp, track.filter.x.copy(), track.filter.P.copy())
            )
            used_c.add(ci)
            used_t.add(ti)

        for ti, track in enumerate(self._tracks):
            if ti not in used_t:
                track.missing += 1

        for ci, cand in enumerate(candidates):
            if ci in used_c:
                continue
            track = TrackedObject(
                track_id=self._next_id,
                filter=_make_filter(cand.position, timestamp, self.config),
                last_update=timestamp,
                pending=True,
                birth_position=cand.position.copy(),
                birth_time=timestamp,
                last_views=cand.views,
                last_reproj_px=cand.mean_reproj_px,
                observations=[
                    TrackPoint(
                        timestamp=timestamp,
                        position=cand.position.copy(),
                        velocity=np.zeros(3),
                        views=cand.view_names(),
                        mean_reproj_px=cand.mean_reproj_px,
                    )
                ],
            )
            track.state_history.append(
                (timestamp, track.filter.x.copy(), track.filter.P.copy())
            )
            self._next_id += 1
            self._tracks.append(track)

        self._tracks = [t for t in self._tracks if t.missing <= self.config.max_missing]
        return FrameFusionResult(
            timestamp=timestamp,
            candidates=candidates,
            observations=observations,
            tracks=list(self._tracks),
            rejected=rejected,
        )
