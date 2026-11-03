"""Базовые типы данных пакета multiview3d.

Здесь собраны простые структуры (dataclass), которые передаются между
модулями: калибровка камеры, 2D-детекции, 3D-наблюдения, треки.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class CameraCalibration:
    """Внутренние и внешние параметры одной камеры.

    Внешние параметры задают переход из мировой системы координат в камеру:
        X_cam = R @ X_world + t

    Единицы ``t`` задаются множителем ``unit``: мировая точка в метрах равна
    ``t * unit`` (``unit=1.0`` — метры, ``unit=0.001`` — если ``t`` записан в
    миллиметрах). Библиотека по умолчанию работает в метрах: калибровка по
    полигону переводит миллиметры в метры, поэтому обычный ``unit == 1.0``.
    """

    name: str
    K: np.ndarray
    dist: np.ndarray
    R: np.ndarray
    t: np.ndarray
    width: int = 0
    height: int = 0
    unit: float = 1.0

    def __post_init__(self) -> None:
        self.K = np.asarray(self.K, dtype=np.float64).reshape(3, 3)
        self.dist = np.asarray(self.dist, dtype=np.float64).ravel()
        if self.dist.size == 0:
            self.dist = np.zeros(5, dtype=np.float64)
        self.R = np.asarray(self.R, dtype=np.float64).reshape(3, 3)
        self.t = np.asarray(self.t, dtype=np.float64).ravel()
        if self.t.size != 3:
            raise ValueError(f"вектор t должен иметь 3 элемента, получено {self.t.size}")
        try:
            self.unit = float(self.unit)
        except (TypeError, ValueError):
            raise ValueError(
                f"unit должен быть числом-множителем в метры (1.0 = метры), получено {self.unit!r}"
            ) from None

    @property
    def projection_matrix(self) -> np.ndarray:
        """Матрица проекции P = K @ [R | t] (в OpenCV 5 нет composeProjectionMatrix)."""
        RT = np.hstack((self.R, self.t.reshape(3, 1)))
        return self.K @ RT

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "K": self.K.tolist(),
            "dist": self.dist.tolist(),
            "R": self.R.tolist(),
            "t": self.t.tolist(),
            "width": int(self.width),
            "height": int(self.height),
            "unit": float(self.unit),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CameraCalibration":
        return cls(
            name=str(data["name"]),
            K=data["K"],
            dist=data.get("dist", []),
            R=data["R"],
            t=data["t"],
            width=int(data.get("width", 0)),
            height=int(data.get("height", 0)),
            unit=float(data.get("unit", 1.0)),
        )


@dataclass
class Detection2D:
    """Детекция объекта на изображении одной камеры.

    ``center`` — координаты центра объекта в пикселях (x, y).
    ``score`` — уверенность детектора в диапазоне [0, 1].
    """

    camera: str
    timestamp: float
    center: np.ndarray
    score: float = 1.0
    bbox: tuple[float, float, float, float] | None = None
    label: str = ""

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=np.float64).ravel()
        if self.center.size != 2:
            raise ValueError("center должен содержать 2 координаты (x, y)")


MEASURED = "MEASURED"
"""3D получен по >=2 видам (триангуляция) — измерение, а не прогноз."""
MONOCULAR = "MONOCULAR"
"""Один вид: метрическая глубина без 3D-модели/опор невозможна (см. §2.6 брифа)."""
PREDICTED_ONLY = "PREDICTED_ONLY"
"""Измерения на кадре нет; позиция — прогноз фильтра Калмана."""
UNOBSERVABLE = "UNOBSERVABLE"
"""Нет валидного 3D-наблюдения и прогноз не выдаётся."""

METHOD_TRIANGULATION = "triangulation"
METHOD_MONOCULAR = "monocular_pnp"
METHOD_PREDICTED = "predicted_only"
METHOD_NONE = "none"


@dataclass
class Observation3D:
    """Триангулированная точка в мировой системе координат (в метрах).

    Поля ``frame``/``method``/``status``/``confidence``/``uncertainty_xyz``
    реализуют DTO из §11 брифа: статус и метод обязательны для корректной
    оценки (нельзя смешивать измерения и прогнозы).
    """

    timestamp: float
    position: np.ndarray
    views: list[str] = field(default_factory=list)
    mean_reproj_px: float = float("nan")
    frame: int | None = None
    method: str = METHOD_TRIANGULATION
    status: str = MEASURED
    confidence: float = float("nan")
    uncertainty_xyz: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64).ravel()
        if self.uncertainty_xyz is not None:
            self.uncertainty_xyz = np.asarray(self.uncertainty_xyz, dtype=np.float64).ravel()

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame,
            "timestamp": float(self.timestamp),
            "position_xyz_m": [float(v) for v in self.position],
            "views": list(self.views),
            "camera_ids": list(self.views),
            "method": self.method,
            "status": self.status,
            "reprojection_error_px": _finite_or_none(self.mean_reproj_px),
            "confidence": _finite_or_none(self.confidence),
            "uncertainty_xyz": (None if self.uncertainty_xyz is None
                                else [float(v) for v in self.uncertainty_xyz]),
        }


@dataclass
class TrackPoint:
    """Точка траектории: сглаженное состояние цели в момент времени."""

    timestamp: float
    position: np.ndarray
    velocity: np.ndarray
    views: list[str] = field(default_factory=list)
    mean_reproj_px: float = float("nan")
    frame: int | None = None
    method: str = METHOD_TRIANGULATION
    status: str = MEASURED
    confidence: float = float("nan")


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


@dataclass
class Track:
    """Трек объекта: идентификатор и хронология точек."""

    track_id: int
    points: list[TrackPoint] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "points": [
                {
                    "timestamp": p.timestamp,
                    "frame_id": p.frame,
                    "position": [float(v) for v in p.position],
                    "velocity": [float(v) for v in p.velocity],
                    "views": list(p.views),
                    "camera_ids": list(p.views),
                    "method": p.method,
                    "status": p.status,
                    "confidence": _finite_or_none(p.confidence),
                    "mean_reproj_px": _finite_or_none(p.mean_reproj_px),
                }
                for p in self.points
            ],
        }
