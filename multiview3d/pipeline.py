"""Конвейер «кадры -> детекции -> 3D-треки»: оркестрация и JSON-конфиг.

Режимы:
* ``MultiViewPipeline.run()`` — реальный времени: фоновые читатели
  ``capture.build_readers`` + ``read_aligned`` по последним кадрам;
* ``MultiViewPipeline.run_offline(sources)`` — покадровый синхронный проход
  (синтетика, файлы видео) через генератор ``capture.read_all_offline``.

История состояний/ковариаций по ``track_id`` собирается здесь и отдаётся в
``kalman.rts_smooth`` для сглаживания траекторий «постфактум».
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .calibration import load_calibration
from .capture import (
    FrameReader,
    FrameSource,
    build_readers,
    read_aligned,
    read_all_offline,
)
from .detect import build_detector
from .fusion import FusionConfig, FrameFusionResult, MultiViewTracker
from .kalman import rts_smooth
from .types import CameraCalibration, Detection2D, Track, TrackPoint


@dataclass
class StreamConfig:
    """Описание одного видеопотока в конфиге."""

    kind: str = "sim"
    """``"sim"`` — синтетика, ``"file"`` — видеофайл, ``"stream"`` — RTSP/USB."""
    source: str | int = ""
    """Путь к файлу / RTSP-адрес / индекс камеры (для file/stream)."""
    name: str = ""
    sim_camera_index: int = 0
    repeat: bool = False
    """Синтетика: зациклить поток. По умолчанию False: при зацикливании камеры
    расходятся по фазе на «wrap» и выравнивание по таймкоду рвётся."""
    detector: dict[str, Any] | None = None
    """Спецификация детектора; None — общий детектор конвейера."""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StreamConfig":
        return cls(
            kind=str(d.get("kind", "sim")),
            source=d.get("source", ""),
            name=str(d.get("name", "")),
            sim_camera_index=int(d.get("sim_camera_index", 0)),
            repeat=bool(d.get("repeat", False)),
            detector=d.get("detector"),
        )


@dataclass
class PipelineConfig:
    """Полный конфиг конвейера: камеры, потоки, детектор, пороги слияния."""

    cameras: list[CameraCalibration] = field(default_factory=list)
    streams: list[StreamConfig] = field(default_factory=list)
    detector: dict[str, Any] = field(default_factory=lambda: {"kind": "blob", "colors": []})
    fusion: FusionConfig = field(default_factory=FusionConfig)
    align_tolerance: float = 0.05
    target_fps: float | None = None
    stall_seconds: float = 1.0
    """``run()`` завершается, если полный кадр не собран дольше этого времени."""
    sim: dict[str, Any] | None = None
    """Параметры ``sim.make_sim_setup``; если задан, камеры берутся из синтетики."""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PipelineConfig":
        fusion_keys = set(FusionConfig.__dataclass_fields__)
        fusion = FusionConfig(**{k: v for k, v in (d.get("fusion") or {}).items()
                                 if k in fusion_keys})
        if d.get("sim"):
            from .sim import make_sim_setup
            cameras = make_sim_setup(**d["sim"]).cameras
        elif d.get("calibration"):
            cameras, _ = load_calibration(d["calibration"])
        elif d.get("cameras"):
            cameras = [CameraCalibration.from_dict(c) for c in d["cameras"]]
        else:
            raise ValueError("конфиг: нужен один из sim / calibration / cameras")
        return cls(
            cameras=cameras,
            streams=[StreamConfig.from_dict(s) for s in d.get("streams", [])],
            detector=d.get("detector", {"kind": "blob", "colors": []}),
            fusion=fusion,
            align_tolerance=float(d.get("align_tolerance", 0.05)),
            target_fps=d.get("target_fps"),
            stall_seconds=float(d.get("stall_seconds", 1.0)),
            sim=d.get("sim"),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "PipelineConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class MultiViewPipeline:
    """Конвейер: синхронные детекции всех камер -> 3D-треки с историей."""

    def __init__(self, cameras: list[CameraCalibration],
                 config: PipelineConfig | None = None) -> None:
        self.cameras = list(cameras)
        self.names = [c.name for c in cameras]
        self.config = config or PipelineConfig(cameras=cameras)
        self.tracker = MultiViewTracker(self.names, self.cameras, self.config.fusion)
        # track_id -> {observations, states, covariances, pos_var, vel_var}
        self.track_history: dict[int, dict[str, Any]] = {}
        self.step_times: list[float] = []
        self.frame_times: list[float] = []
        self.last_result: FrameFusionResult | None = None

    # ---------- сборка источников из конфига ----------

    def build_sources(self, *, repeat: bool | None = None) -> tuple[list[FrameSource], Any]:
        """Источники кадров по ``config.streams``; второй элемент — синтетика или None.

        ``repeat`` — переопределить зацикливание синтетики (офлайн-проход: False).
        """
        dataset = None
        if self.config.sim:
            from .sim import make_sim_setup
            dataset = make_sim_setup(**self.config.sim)
        sources: list[FrameSource] = []
        for sc in self.config.streams:
            if sc.kind == "sim":
                if dataset is None:
                    raise ValueError("stream kind=sim, но в конфиге нет секции sim")
                spec = sc.detector or self.config.detector
                loop = sc.repeat if repeat is None else repeat
                if spec.get("kind") == "oracle":
                    from .sim import OracleSimSource
                    src: FrameSource = OracleSimSource(dataset, sc.sim_camera_index,
                                                       repeat=loop)
                else:
                    from .sim import SimCameraStream, SimSource
                    src = SimSource(SimCameraStream(dataset, sc.sim_camera_index,
                                                    repeat=loop),
                                    build_detector(spec))
                if sc.name:
                    src.name = sc.name
                sources.append(src)
            elif sc.kind in ("file", "stream"):
                from .capture import OpenCvFrameSource
                sources.append(OpenCvFrameSource(sc.source))
            else:
                raise ValueError(f"неизвестный kind потока: {sc.kind!r}")
        return sources, dataset

    # ---------- шаги ----------

    def step(self, timestamp: float,
             detections_by_name: dict[str, list[Detection2D]]) -> FrameFusionResult:
        """Один кадр: детекции по имени камеры -> слияние/трекинг; обновляет историю."""
        detections = [detections_by_name.get(name, []) for name in self.names]
        t0 = time.perf_counter()
        result = self.tracker.step(timestamp, detections)
        self.step_times.append(time.perf_counter() - t0)
        self.frame_times.append(timestamp)
        for track in result.tracks:
            self.track_history[track.track_id] = {
                "observations": list(track.observations),
                "states": [s for _, s, _ in track.state_history],
                "covariances": [p for _, _, p in track.state_history],
                "pos_var": track.filter.pos_var,
                "vel_var": track.filter.vel_var,
            }
        self.last_result = result
        return result

    def _detections_from_aligned(self, aligned: dict[Any, tuple]) -> dict[str, list[Detection2D]]:
        """Из выровненных кадров (имя или индекс) -> детекции по именам камер."""
        out: dict[str, list[Detection2D]] = {}
        for key, item in aligned.items():
            if isinstance(key, int):
                name = self.names[key]
            else:
                name = key if key in self.names else self.names[int(key)]
            out[name] = list(item[2] or [])
        return out

    def run_offline(self, sources: list[FrameSource], *, max_frames: int | None = None,
                    align_tolerance: float | None = None) -> list[Track]:
        """Покадровый офлайн-проход; возвращает сырые треки (до RTS-сглаживания)."""
        tol = self.config.align_tolerance if align_tolerance is None else align_tolerance
        n = 0
        for aligned in read_all_offline(sources, align_tolerance=tol):
            timestamp = float(aligned[0][0])
            self.step(timestamp, self._detections_from_aligned(aligned))
            n += 1
            if max_frames is not None and n >= max_frames:
                break
        return self.raw_tracks()

    def run(self, sources: list[FrameSource], *, max_frames: int | None = None,
            readers: list[FrameReader] | None = None) -> list[Track]:
        """Реальный времени: фоновые читатели, выравнивание по таймкоду, темп target_fps.

        Завершается по ``max_frames``; для бесконечного live-цикла используй
        ``step_from_readers`` снаружи. Читатели закрываются в ``finally``.
        """
        if readers is None:
            readers = build_readers(sources, self.names[:len(sources)])
        n = 0
        stall = float(self.config.stall_seconds)
        t_last_step = time.perf_counter()
        try:
            while True:
                t0 = time.perf_counter()
                aligned = read_aligned(readers, align_tolerance=self.config.align_tolerance,
                                      timeout=0.1)
                if aligned is None:
                    # Стоп-защита: источники исчерпаны или камеры молчат дольше stall.
                    if all(r.closed for r in readers) or (time.perf_counter() - t_last_step) > stall:
                        break
                    time.sleep(0.01)
                    continue
                t_last_step = time.perf_counter()
                timestamp = max(item[0] for item in aligned.values())
                self.step(timestamp, self._detections_from_aligned(aligned))
                n += 1
                if max_frames is not None and n >= max_frames:
                    break
                if self.config.target_fps:
                    pause = 1.0 / self.config.target_fps - (time.perf_counter() - t0)
                    if pause > 0:
                        time.sleep(pause)
        finally:
            for r in readers:
                r.close()
        return self.raw_tracks()

    # ---------- треки и сглаживание ----------

    def raw_tracks(self) -> list[Track]:
        """Треки по сырым (отфильтрованным Калманом, не сглаженным) точкам."""
        return [Track(tid, list(h["observations"]))
                for tid, h in sorted(self.track_history.items()) if h["observations"]]

    def smoothed_tracks(self) -> list[Track]:
        """Треки после RTS-сглаживания истории состояний каждого трека."""
        out: list[Track] = []
        for tid, h in sorted(self.track_history.items()):
            obs = h["observations"]
            if not obs:
                continue
            times = [p.timestamp for p in obs]
            smoothed = rts_smooth(times, np.asarray(h["states"]),
                                  np.asarray(h["covariances"]),
                                  h["pos_var"], h["vel_var"])
            points = [TrackPoint(s.timestamp, s.position, s.velocity,
                                 obs[i].views, obs[i].mean_reproj_px)
                      for i, s in enumerate(smoothed)]
            out.append(Track(tid, points))
        return out

    def stats(self, readers: list[FrameReader] | None = None) -> dict[str, Any]:
        """Частота обработки, время шага, сводка читателей (для live-режима)."""
        n = len(self.step_times)
        total = sum(self.step_times)
        st: dict[str, Any] = {
            "frames": n,
            "processing_fps": (n / total) if total > 1e-9 else 0.0,
            "step_ms_mean": (1000.0 * total / n) if n else 0.0,
            "step_ms_max": (1000.0 * max(self.step_times)) if n else 0.0,
        }
        if readers:
            st["readers"] = {
                r.name: {
                    "frames_read": r.stats.frames_read,
                    "frames_dropped": r.stats.frames_dropped,
                    "mean_latency_ms": r.stats.mean_latency_ms,
                }
                for r in readers
            }
        return st
