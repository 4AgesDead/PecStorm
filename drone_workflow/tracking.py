
"""3D multi-camera drone tracking adapter for PecStorm.

На входе:
    detections_by_frame[frame][camera_name] -> list[Detection2D]

Предполагается, что камеры уже откалиброваны в общей мировой системе,
а наблюдения синхронизированы по времени.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Mapping, Sequence

from multiview3d.fusion import (
    FrameFusionResult,
    FusionConfig,
    MultiViewTracker,
    TrackedObject,
)
from multiview3d.types import CameraCalibration, Detection2D


@dataclass
class DroneTrackingResult:
    """Результат одного offline-прохода трекера."""

    camera_names: list[str]
    frame_results: list[FrameFusionResult]
    tracks: list[TrackedObject]

    @property
    def frames_processed(self) -> int:
        return len(self.frame_results)

    @property
    def measured_points(self) -> int:
        return sum(len(track.observations) for track in self.tracks)

    @property
    def predicted_points(self) -> int:
        return sum(
            sum(point.status == "PREDICTED_ONLY" for point in track.points)
            for track in self.tracks
        )


def _resolve_timestamp(
    frame: int,
    by_camera: Mapping[str, Sequence[Detection2D]],
    *,
    frame_times: Mapping[int, float] | None,
    fps: float | None,
    first_frame: int,
    timestamp_tolerance: float,
) -> float:
    """Определяет единый timestamp кадра и проверяет синхронизацию."""

    if frame_times is not None and frame in frame_times:
        timestamp = float(frame_times[frame])
    else:
        observed_times = [
            float(det.timestamp)
            for detections in by_camera.values()
            for det in detections
        ]

        if observed_times:
            lo = min(observed_times)
            hi = max(observed_times)

            if hi - lo > timestamp_tolerance:
                raise ValueError(
                    f"Frame {frame} is not synchronized: "
                    f"timestamp spread={hi - lo:.6f}s, "
                    f"tolerance={timestamp_tolerance:.6f}s"
                )

            timestamp = float(median(observed_times))

        elif fps is not None:
            timestamp = (frame - first_frame) / float(fps)

        else:
            raise ValueError(
                f"Frame {frame} has no detections. "
                "Provide frame_times or fps."
            )

    if not math.isfinite(timestamp):
        raise ValueError(f"Invalid timestamp for frame {frame}")

    return timestamp


def track_synchronized_frames(
    cameras: Sequence[CameraCalibration],
    detections_by_frame: Mapping[
        int, Mapping[str, Sequence[Detection2D]]
    ],
    *,
    fusion: FusionConfig | None = None,
    frame_times: Mapping[int, float] | None = None,
    fps: float | None = None,
    first_frame: int = 0,
    timestamp_tolerance: float = 0.02,
) -> DroneTrackingResult:
    """Выполняет многокамерный 3D-трекинг дрона или нескольких объектов.

    Порядок cameras определяет порядок списков детекций, передаваемых
    в MultiViewTracker. Отсутствующее наблюдение задаётся пустым списком.

    Для кадра без единой детекции требуется frame_times или fps.
    """

    if not cameras:
        raise ValueError("At least one calibrated camera is required")

    if not detections_by_frame:
        raise ValueError("detections_by_frame is empty")

    if timestamp_tolerance < 0 or not math.isfinite(timestamp_tolerance):
        raise ValueError("timestamp_tolerance must be finite and >= 0")

    if fps is not None and (not math.isfinite(fps) or fps <= 0):
        raise ValueError("fps must be positive and finite")

    camera_names = [str(camera.name) for camera in cameras]

    if len(set(camera_names)) != len(camera_names):
        raise ValueError("Camera names must be unique")

    known_cameras = set(camera_names)

    unknown = {
        str(name)
        for per_camera in detections_by_frame.values()
        for name in per_camera
        if str(name) not in known_cameras
    }

    if unknown:
        raise ValueError(f"Unknown cameras in detections: {sorted(unknown)}")

    tracker = MultiViewTracker(
        camera_names=camera_names,
        cameras=list(cameras),
        config=fusion or FusionConfig(association_mode="epipolar"),
    )

    frame_results: list[FrameFusionResult] = []
    all_tracks: dict[int, TrackedObject] = {}
    previous_timestamp: float | None = None

    for frame_value in sorted(detections_by_frame):
        frame = int(frame_value)
        by_camera = detections_by_frame[frame_value]

        timestamp = _resolve_timestamp(
            frame,
            by_camera,
            frame_times=frame_times,
            fps=fps,
            first_frame=first_frame,
            timestamp_tolerance=timestamp_tolerance,
        )

        if (
            previous_timestamp is not None
            and timestamp <= previous_timestamp
        ):
            raise ValueError(
                f"Timestamps must increase: frame {frame}, "
                f"time={timestamp:.6f}, previous={previous_timestamp:.6f}"
            )

        previous_timestamp = timestamp

        # Собираем наблюдения в том же порядке, что и cameras.
        per_camera: list[list[Detection2D]] = []

        for camera_name in camera_names:
            detections = list(by_camera.get(camera_name, ()))

            for detection in detections:
                if detection.camera != camera_name:
                    raise ValueError(
                        f"Detection is grouped under {camera_name!r}, "
                        f"but declares camera={detection.camera!r}"
                    )

            per_camera.append(detections)

        frame_result = tracker.step(
            timestamp=timestamp,
            detections=per_camera,
            frame=frame,
        )

        frame_results.append(frame_result)

        # Сохраняем ссылки и на треки, которые позднее станут неактивными.
        for track in frame_result.tracks:
            all_tracks[track.track_id] = track

    return DroneTrackingResult(
        camera_names=camera_names,
        frame_results=frame_results,
        tracks=[
            all_tracks[track_id]
            for track_id in sorted(all_tracks)
        ],
    )


def write_tracks_csv(
    result: DroneTrackingResult,
    path: str | Path,
    *,
    include_predicted: bool = True,
) -> int:
    """Сохраняет историю 3D-треков в CSV."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "track_id",
        "frame",
        "timestamp",
        "x",
        "y",
        "z",
        "vx",
        "vy",
        "vz",
        "status",
        "method",
        "n_views",
        "views",
        "mean_reproj_px",
        "confidence",
    ]

    count = 0

    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()

        for track in result.tracks:
            for point in track.points:
                if (
                    not include_predicted
                    and point.status == "PREDICTED_ONLY"
                ):
                    continue

                x, y, z = map(float, point.position)
                vx, vy, vz = map(float, point.velocity)

                writer.writerow({
                    "track_id": track.track_id,
                    "frame": (
                        "" if point.frame is None else point.frame
                    ),
                    "timestamp": f"{point.timestamp:.9f}",
                    "x": f"{x:.9f}",
                    "y": f"{y:.9f}",
                    "z": f"{z:.9f}",
                    "vx": f"{vx:.9f}",
                    "vy": f"{vy:.9f}",
                    "vz": f"{vz:.9f}",
                    "status": point.status,
                    "method": point.method,
                    "n_views": len(point.views),
                    "views": "|".join(point.views),
                    "mean_reproj_px": _finite_text(
                        point.mean_reproj_px
                    ),
                    "confidence": _finite_text(point.confidence),
                })

                count += 1

    return count



def write_tracks_json(
    result: DroneTrackingResult,
    path: str | Path,
) -> None:
    """Сохраняет полную историю 3D-треков в JSON."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    tracks_data = []

    for track in result.tracks:
        points_data = []

        for point in track.points:
            points_data.append({
                "timestamp": float(point.timestamp),
                "frame": point.frame,
                "position_xyz_m": [
                    float(v) for v in point.position
                ],
                "velocity_xyz_mps": [
                    float(v) for v in point.velocity
                ],
                "views": list(point.views),
                "status": point.status,
                "method": point.method,
                "mean_reproj_px": _finite_or_none(
                    point.mean_reproj_px
                ),
                "confidence": _finite_or_none(
                    point.confidence
                ),
            })

        tracks_data.append({
            "track_id": int(track.track_id),
            "points": points_data,
        })

    payload = {
        "camera_names": result.camera_names,
        "frames_processed": result.frames_processed,
        "track_count": len(result.tracks),
        "measured_points": result.measured_points,
        "predicted_points": result.predicted_points,
        "tracks": tracks_data,
    }

    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _finite_or_none(value: float) -> float | None:
    """Заменяет NaN и Infinity на JSON null."""
    value = float(value)
    return value if math.isfinite(value) else None



def _finite_text(value: float) -> str:
    value = float(value)
    return f"{value:.9f}" if math.isfinite(value) else ""
