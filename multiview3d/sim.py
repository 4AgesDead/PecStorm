"""Синтетические калиброванные камеры и видеопотоки.

Назначение: численная проверка точности 3D-координат в миллиметрах. Камеры имеют
известные K, R, t и коэффициенты искажения, кадры рендерятся точной моделью
проекции, поэтому ground truth 3D-траекторий известен аналитически.

Важно: синтетика не заменяет реальность. Она проверяет геометрию, сопоставление
видов и фильтр Калмана, но не проверяет качество реального детектора.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from .camera import CameraCalibration, project_points
from .detect import Detection2D, Detector, OracleDetector
from .types import TrackPoint

ObjectPath = Callable[[float], np.ndarray]


def look_at_rotation(origin: np.ndarray, target: np.ndarray,
                     up: np.ndarray | None = None) -> np.ndarray:
    """R world->camera для камеры в origin, направленной на target."""
    origin = np.asarray(origin, dtype=float)
    forward = np.asarray(target, dtype=float) - origin
    forward = forward / np.linalg.norm(forward)
    up = np.array([0.0, 0.0, 1.0]) if up is None else np.asarray(up, dtype=float)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.vstack([right, down, forward])


@dataclass
class SimDataset:
    """Синтетическая сцена: камеры, пути объектов, расписание кадров."""

    cameras: list[CameraCalibration]
    paths: list[ObjectPath]
    frame_times: list[float]
    labels: list[str]
    fps: float
    unit: float = 1.0
    ball_radius: float = 0.08
    ball_colors: list[tuple[int, int, int]] = field(
        default_factory=lambda: [(70, 60, 230), (60, 200, 60), (40, 180, 230), (200, 60, 200)]
    )

    @property
    def frame_count(self) -> int:
        return len(self.frame_times)

    def truth_at(self, t: float) -> list[np.ndarray]:
        """Ground truth 3D-позиции всех объектов в момент t."""
        return [np.asarray(path(t), dtype=float) for path in self.paths]

    def ground_truth_tracks(self) -> list[list[TrackPoint]]:
        """Эталонные траектории по всем кадрам (без наблюдений)."""
        tracks: list[list[TrackPoint]] = []
        for k in range(len(self.paths)):
            pts = [TrackPoint(t, self.truth_at(t)[k], np.zeros(3), [], 0.0)
                   for t in self.frame_times]
            tracks.append(pts)
        return tracks

    def camera_frame(self, camera_index: int, t: float) -> np.ndarray:
        """Рендерит кадр камеры в момент t (точное отображение сфер в пиксели)."""
        cam = self.cameras[camera_index]
        frame = np.zeros((cam.height, cam.width, 3), dtype=np.uint8)
        frame[:, :] = (36, 30, 26)
        for y in range(0, cam.height, 48):
            cv2.line(frame, (0, y), (cam.width, y), (54, 48, 42), 1)
        for x in range(0, cam.width, 64):
            cv2.line(frame, (x, 0), (x, cam.height), (54, 48, 42), 1)

        points = self.truth_at(t)
        pixels = project_points(cam, points)
        order = np.argsort([np.linalg.norm(p - cam_center(cam)) for p in points])
        for k in order:
            u, v = pixels[k]
            if not (0 <= u < cam.width and 0 <= v < cam.height):
                continue
            depth = float(np.linalg.norm(points[k] - cam_center(cam)))
            radius = max(4, int(round(cam.K[0, 0] * self.ball_radius / max(depth, 1e-6))))
            color = self.ball_colors[k % len(self.ball_colors)]
            cv2.circle(frame, (int(round(u)), int(round(v))), radius, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (int(round(u)), int(round(v))), radius, (255, 255, 255), 1, cv2.LINE_AA)
        return frame


def cam_center(cam: CameraCalibration) -> np.ndarray:
    """Центр проекции камеры в мировых координатах: C = -R^T t."""
    r = np.asarray(cam.R, dtype=float)
    return -r.T @ np.asarray(cam.t, dtype=float)


class SimCameraStream:
    """FrameSource: синтетический видеопоток одной калиброванной камеры."""

    def __init__(self, dataset: SimDataset, camera_index: int, *, repeat: bool = False) -> None:
        self.dataset = dataset
        self.camera_index = camera_index
        self.name = dataset.cameras[camera_index].name
        self._pos = 0
        self.repeat = repeat
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        """Ресурсов нет; флаг нужен для протокола ``capture.FrameSource``."""
        self._closed = True

    def read(self) -> tuple[float, np.ndarray, None] | None:
        if self._closed or self._pos >= self.dataset.frame_count:
            if not self.repeat or self._closed:
                return None
            self._pos = 0
        t = self.dataset.frame_times[self._pos]
        frame = self.dataset.camera_frame(self.camera_index, t)
        self._pos += 1
        return t, frame, None


class SimSource:
    """FrameSource: синтетический поток + детектор 2D-объектов."""

    name: str = "sim"

    def __init__(self, stream: SimCameraStream, detector: Detector) -> None:
        self.stream = stream
        self.detector = detector
        self.name = stream.name
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        """Закрывает поток; сам источник ресурсов не держит."""
        self._closed = True
        self.stream.close()

    def read(self) -> tuple[float, np.ndarray, list[Detection2D]] | None:
        if self._closed:
            return None
        item = self.stream.read()
        if item is None:
            return None
        t, frame, _ = item
        return t, frame, self.detector.detect(frame)


class OracleSimSource:
    """FrameSource: синтетический поток с «идеальными» детекциями по ground truth.

    Перед каждым ``detect()`` проецирует эталонные 3D-точки точной моделью
    камеры, поэтому детекции — истинные пиксели без шума детектора. Только для
    численной валидации геометрии и трекинга.
    """

    def __init__(self, dataset: SimDataset, camera_index: int, *, repeat: bool = False) -> None:
        self.dataset = dataset
        self.stream = SimCameraStream(dataset, camera_index, repeat=repeat)
        self.name = self.stream.name
        self.detector = OracleDetector()
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        self._closed = True
        self.stream.close()

    def read(self) -> tuple[float, np.ndarray, list[Detection2D]] | None:
        if self._closed:
            return None
        item = self.stream.read()
        if item is None:
            return None
        t, frame, _ = item
        pixels = project_points(self.dataset.cameras[self.stream.camera_index],
                                 self.dataset.truth_at(t))
        gt = {
            k: (float(px[0]), float(px[1]))
            for k, px in enumerate(pixels)
            if np.isfinite(px).all()
        }
        self.detector.set_ground_truth(gt)
        return t, frame, self.detector.detect(frame)


def make_sim_setup(
    *,
    num_cameras: int = 3,
    num_objects: int = 2,
    width: int = 1280,
    height: int = 720,
    fps: float = 25.0,
    duration_s: float = 6.0,
    workspace: tuple[float, float, float, float, float, float] = (-2.0, 2.0, -2.0, 2.0, 0.0, 3.0),
    camera_ring_radius: float = 5.0,
    camera_height: float = 2.2,
    ball_radius: float = 0.08,
    fx: float = 1100.0,
    distortion: tuple[float, ...] = (-0.25, 0.09, 0.001, 0.002, -0.02),
    seed: int = 7,
    unit: float = 1.0,
) -> SimDataset:
    """Строит синтетическую сцену с точно известной калибрацией камер."""
    rng = np.random.default_rng(seed)
    target = np.array([0.0, 0.0, 1.0])

    cameras: list[CameraCalibration] = []
    for i in range(num_cameras):
        angle = 2.0 * np.pi * i / num_cameras
        origin = np.array([camera_ring_radius * np.cos(angle),
                           camera_ring_radius * np.sin(angle), camera_height])
        rot = look_at_rotation(origin, target)
        cx, cy = width / 2.0, height / 2.0
        calib = CameraCalibration(
            name=f"cam{i + 1}",
            K=np.array([[fx, 0.0, cx], [0.0, fx, cy], [0.0, 0.0, 1.0]]),
            dist=np.array(distortion, dtype=float),
            R=rot,
            t=-rot @ origin,
            width=width,
            height=height,
            unit=1.0,
        )
        cameras.append(calib)

    lo = np.array(workspace[:3])
    hi = np.array(workspace[3:])
    paths: list[ObjectPath] = []
    for k in range(num_objects):
        base = lo + (hi - lo) * rng.uniform(0.25, 0.75, size=3)
        base[2] = max(base[2], workspace[2] + 0.3)
        amp = (hi - lo) * rng.uniform(0.10, 0.22, size=3)
        freq = rng.uniform(0.25, 0.7, size=3)
        phase = rng.uniform(0.0, 2.0 * np.pi, size=3)

        def path(t: float, base=base, amp=amp, freq=freq, phase=phase) -> np.ndarray:
            return base + amp * np.sin(2.0 * np.pi * freq * t + phase)

        paths.append(path)

    frame_times = [i / fps for i in range(int(round(duration_s * fps)))]
    labels = [f"ball{k + 1}" for k in range(num_objects)]
    return SimDataset(cameras=cameras, paths=paths, frame_times=frame_times, labels=labels,
                      fps=fps, unit=unit, ball_radius=ball_radius)
