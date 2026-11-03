"""Позы камер по номеру кадра (CameraStateProvider).

Для датасета «Пеленга»: камеры 1–4 статичны, камера 5 меняет yaw по кадрам.
Provider кэширует строки CSV и строит ``CameraCalibration`` по запросу;
статичные камеры возвращают один и тот же неизменяемый объект.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .types import CameraCalibration
from .unreal_camera import UnrealRow, camera_from_row, parse_camera_csv


@dataclass
class CameraSpec:
    name: str
    csv_path: str
    width: int
    height: int


class CameraStateProvider:
    """Календарь поз: ``calibration(name, frame)`` и ``calibrations_at(frame)``.

    Поза камеры 5 меняется по кадру; сырые углы не интерполируются линейно —
    матрица поворота строится из угла кадра напрямую (без ±180° скачков).
    """

    def __init__(self, specs: list[CameraSpec], image_size_fallback: tuple[int, int] | None = None):
        self._specs: dict[str, CameraSpec] = {}
        self._rows: dict[str, dict[int, UnrealRow]] = {}
        self._static: dict[str, CameraCalibration] = {}
        self._meta: dict[str, tuple] = {}
        self._cache: dict[tuple[str, int], CameraCalibration] = {}
        for spec in specs:
            parsed = parse_camera_csv(spec.csv_path)
            if parsed.focal_mm is None or parsed.sensor_mm is None:
                raise ValueError(f"в {spec.csv_path} нет focal length / размера сенсора")
            w = spec.width or (image_size_fallback[0] if image_size_fallback else 1920)
            h = spec.height or (image_size_fallback[1] if image_size_fallback else 1080)
            self._specs[spec.name] = spec
            self._rows[spec.name] = {r.frame: r for r in parsed.rows}
            self._meta[spec.name] = (parsed.focal_mm, parsed.sensor_mm, w, h)
            rot_keys = {r.rot for r in parsed.rows}
            loc_keys = {r.loc for r in parsed.rows}
            if len(rot_keys) == 1 and len(loc_keys) == 1:
                self._static[spec.name] = camera_from_row(
                    parsed.rows[0], spec.name, parsed.focal_mm, parsed.sensor_mm, w, h)

    @property
    def names(self) -> list[str]:
        return sorted(self._specs)

    @property
    def specs(self) -> list[CameraSpec]:
        return [self._specs[n] for n in sorted(self._specs)]

    def is_static(self, name: str) -> bool:
        return name in self._static

    def frames_available(self, name: str) -> list[int]:
        return sorted(self._rows[name])

    def focal_and_sensor(self, name: str) -> tuple[float, tuple[float, float]]:
        """(focal_mm, (sensor_w_mm, sensor_h_mm)) из метаданных CSV — для диагностики intrinsics."""
        focal_mm, sensor_mm, _, _ = self._meta[name]
        return focal_mm, sensor_mm

    def nearest_frame(self, name: str, frame: int) -> int:
        """Ближайший кадр, для которого в CSV есть строка (без интерполяции углов)."""
        frames = self._rows[name]
        if frame in frames:
            return frame
        keys = sorted(frames)
        if not keys:
            raise KeyError(f"в CSV камеры {name} нет строк")
        pos = int(np.searchsorted(keys, frame))
        candidates = [keys[min(pos, len(keys) - 1)], keys[max(pos - 1, 0)]]
        return min(candidates, key=lambda k: abs(k - frame))

    def calibration(self, name: str, frame: int) -> CameraCalibration:
        if name in self._static:
            return self._static[name]
        key = (name, frame)
        cam = self._cache.get(key)
        if cam is None:
            row = self._rows[name][frame]
            focal_mm, sensor_mm, w, h = self._meta[name]
            cam = camera_from_row(row, name, focal_mm, sensor_mm, w, h)
            self._cache[key] = cam
        return cam

    def calibrations_at(self, frame: int, *, nearest: bool = False) -> dict[str, CameraCalibration]:
        """Позы камер на кадре ``frame``.

        ``nearest=False`` — только камеры, у которых есть строка ровно на этом
        кадре (строгая синхронизация). ``nearest=True`` — для камер без строки
        берётся ближайший кадр CSV (без интерполяции углов; факт подмены
        кадра вызывающий код обязан учитывать).
        """
        out: dict[str, CameraCalibration] = {}
        for name in self._specs:
            rows = self._rows[name]
            if frame in rows:
                out[name] = self.calibration(name, frame)
            elif nearest:
                f = self.nearest_frame(name, frame)
                out[name] = self.calibration(name, f)
        return out
