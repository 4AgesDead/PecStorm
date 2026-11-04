"""Датасет «Пеленга 2» (scene2): 9 камер, без GT, движущаяся cam2.

Формат CSV камер:
``Frame;Loc_X;Loc_Y;Loc_Z;Rot_Roll;Rot_Pitch;Rot_Yaw;Focal_Length_mm``
(разделитель ``;``, сантиметры UE, углы в градусах). Размер сенсора лежит
отдельно в ``matrix_size.txt`` (``23,76 mm * 13,365 mm``).

GT в наборе нет — только камеры и видео. Поэтому класс не требует GT и умеет
отдавать калибровки по кадрам; оценка качества — только self-consistency
(перепроекция) или скрытый GT организаторов.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .types import CameraCalibration
from .unreal_camera import UnrealRow, camera_from_row, parse_camera_csv

SENSOR_RE = re.compile(r"([\d.,]+)\s*mm\s*\*\s*([\d.,]+)\s*mm", re.IGNORECASE)


def read_sensor_mm(path: str | Path) -> tuple[float, float]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    m = SENSOR_RE.search(text)
    if not m:
        raise ValueError(f"не найден размер сенсора в {path}: {text!r}")
    return float(m.group(1).replace(",", ".")), float(m.group(2).replace(",", "."))


def read_focal_mm(path: str | Path) -> float:
    with Path(path).open(encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=";")
        first = next(reader, None)
        if not first:
            raise ValueError(f"пустой CSV камеры: {path}")
        value = (first.get("Focal_Length_mm") or "").strip()
        if not value:
            raise ValueError(f"нет Focal_Length_mm в {path}")
        return float(value.replace(",", "."))


def video_props(path: str | Path) -> tuple[int, int, float, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"не открыть видео: {path}")
    props = (int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))),
             int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
             float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
             int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
    cap.release()
    return props


@dataclass
class Scene2Camera:
    name: str
    index: int
    video: Path
    csv: Path


class Scene2Provider:
    """Калибровки 9 камер по кадрам; cam2 движется, остальные статичны."""

    def __init__(self, specs: list[Scene2Camera], sensor_mm: tuple[float, float]):
        self._sensor = sensor_mm
        self._rows: dict[str, dict[int, UnrealRow]] = {}
        self._focal: dict[str, float] = {}
        self._size: dict[str, tuple[int, int]] = {}
        self._static: dict[str, CameraCalibration] = {}
        self._cache: dict[tuple[str, int], CameraCalibration] = {}

        for spec in specs:
            parsed = parse_camera_csv(spec.csv)
            if not parsed.rows:
                raise ValueError(f"нет строк в {spec.csv}")
            rows = {r.frame: r for r in parsed.rows}
            focal = read_focal_mm(spec.csv)
            w, h, _, _ = video_props(spec.video)
            self._rows[spec.name] = rows
            self._focal[spec.name] = focal
            self._size[spec.name] = (w, h)
            locs = {r.loc for r in parsed.rows}
            rots = {r.rot for r in parsed.rows}
            if len(locs) == 1 and len(rots) == 1:
                self._static[spec.name] = camera_from_row(
                    parsed.rows[0], spec.name, focal, sensor_mm, w, h)

    @property
    def names(self) -> list[str]:
        return sorted(self._rows)

    @property
    def sensor_mm(self) -> tuple[float, float]:
        return self._sensor

    def is_static(self, name: str) -> bool:
        return name in self._static

    def frames_available(self, name: str) -> list[int]:
        return sorted(self._rows[name])

    def calibration(self, name: str, frame: int) -> CameraCalibration:
        if name in self._static:
            return self._static[name]
        key = (name, frame)
        cam = self._cache.get(key)
        if cam is None:
            row = self._rows[name][frame]
            w, h = self._size[name]
            cam = camera_from_row(row, name, self._focal[name], self._sensor, w, h)
            self._cache[key] = cam
        return cam

    def calibrations_at(self, frame: int, nearest: bool = False
                        ) -> dict[str, CameraCalibration]:
        out: dict[str, CameraCalibration] = {}
        for name, rows in self._rows.items():
            if frame in rows:
                out[name] = self.calibration(name, frame)
            elif nearest:
                keys = sorted(rows)
                pos = int(np.searchsorted(keys, frame))
                cand = [keys[min(pos, len(keys) - 1)], keys[max(pos - 1, 0)]]
                out[name] = self.calibration(name, min(cand, key=lambda k: abs(k - frame)))
        return out


@dataclass
class Scene2Dataset:
    root: Path
    cameras: list[Scene2Camera]
    sensor_mm: tuple[float, float]
    fps: float = 30.0
    _provider: Scene2Provider | None = field(default=None, repr=False)

    @classmethod
    def discover(cls, root: str | Path = "scene2") -> "Scene2Dataset":
        root = Path(root)
        sensor = read_sensor_mm(root / "matrix_size.txt")
        cameras: list[Scene2Camera] = []
        for i in range(1, 10):
            video = root / f"Way2.FinalImage.CineCameraActor{i}.mp4"
            csv_path = root / f"object_transform_CineCameraActor{i}.csv"
            if not video.exists() or not csv_path.exists():
                raise FileNotFoundError(f"нет файлов камеры {i}: {video} / {csv_path}")
            cameras.append(Scene2Camera(f"cam{i}", i, video, csv_path))
        ds = cls(root=root, cameras=cameras, sensor_mm=sensor)
        ds._provider = Scene2Provider(cameras, sensor)
        ds.fps = video_props(cameras[0].video)[2]
        return ds

    @property
    def provider(self) -> Scene2Provider:
        assert self._provider is not None
        return self._provider

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.cameras]

    def frame_count(self) -> int:
        return min(video_props(c.video)[3] for c in self.cameras)

    def common_frame_range(self) -> tuple[int, int]:
        return 0, self.frame_count() - 1

    def calibration(self, name: str, frame: int) -> CameraCalibration:
        return self.provider.calibration(name, frame)

    def ground_truth_by_frame(self) -> dict[int, np.ndarray]:
        """GT в scene2 нет; метод-заглушка для совместимости с пультом."""
        return {}
