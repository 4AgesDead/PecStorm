"""Конвейер «кадры -> детекции -> 3D-треки»: оркестрация и JSON-конфиг.

Режимы:
* ``MultiViewPipeline.run()`` — реальный времени: фоновые читатели
  ``capture.build_readers`` + ``read_aligned`` по последним кадрам (частичная
  доступность камер поддержана: ``allow_partial``/``min_cameras``);
* ``MultiViewPipeline.run_offline(sources)`` — покадровый синхронный проход
  (синтетика, файлы видео, датасет «Пеленга»).

Синхронизация:
* если задан ``fps``, номер кадра считается как ``round(timestamp * fps)``;
* если подключён ``CameraStateProvider`` (датасет «Пеленга»), позы камер
  обновляются на **каждом** кадре (динамическая cam5), а не один раз при
  создании трекера — дефект P0.5 брифа.

История состояний/ковариаций по ``track_id`` собирается здесь и отдаётся в
``kalman.rts_smooth`` для сглаживания «постфактум». Кроме измеренных точек
(``observations``) трек хранит все точки (``points``), включая прогнозы со
статусом ``PREDICTED_ONLY`` — это нужно для coverage-метрик §7.6 брифа.
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
from .sources import DetectingFrameSource
from .types import CameraCalibration, Detection2D, MEASURED, PREDICTED_ONLY, Track, TrackPoint


@dataclass
class StreamConfig:
    """Описание одного видеопотока в конфиге."""

    kind: str = "sim"
    """``"sim"`` — синтетика, ``"file"`` — видеофайл, ``"stream"`` — RTSP/USB,
    ``"truck"`` — камера датасета «Пеленга» (``name`` = cam1..cam5)."""
    source: str | int = ""
    """Путь к файлу / RTSP-адрес / индекс камеры (для file/stream)."""
    name: str = ""
    sim_camera_index: int = 0
    repeat: bool = False
    """Синтетика: зациклить поток. По умолчанию False: при зацикливании камеры
    расходятся по фазе на «wrap» и выравнивание по таймкоду рвётся."""
    detector: dict[str, Any] | None = None
    """Спецификация детектора; None — общий детектор конвейера."""
    frame_start: int | None = None
    """Первый кадр (для ``kind=truck``); None — общая область датасета."""
    frame_end: int | None = None
    """Последний кадр включительно (для ``kind=truck``)."""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StreamConfig":
        return cls(
            kind=str(d.get("kind", "sim")),
            source=d.get("source", ""),
            name=str(d.get("name", "")),
            sim_camera_index=int(d.get("sim_camera_index", 0)),
            repeat=bool(d.get("repeat", False)),
            detector=d.get("detector"),
            frame_start=d.get("frame_start"),
            frame_end=d.get("frame_end"),
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
    fps: float | None = None
    """Частота кадров для пересчёта «таймкод <-> номер кадра» (для truck — 30)."""
    frame_start: int = 0
    """Первый обрабатываемый кадр (общая область датасета)."""
    frame_end: int | None = None
    """Последний обрабатываемый кадр включительно."""
    truck: dict[str, Any] | None = None
    """Секция датасета «Пеленга»: ``{"root": ..., "repairs": ...}``."""
    truck_dataset: Any = None
    """``TruckDataset`` — создаётся в ``from_dict``, если задана секция truck."""
    partial_cameras: bool = True
    """Live-режим: разрешать шаг при неполном наборе камер (>= min_cameras_live)."""
    min_cameras_live: int = 1

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PipelineConfig":
        fusion_keys = set(FusionConfig.__dataclass_fields__)
        fusion = FusionConfig(**{k: v for k, v in (d.get("fusion") or {}).items()
                                 if k in fusion_keys})
        dataset = None
        frame_start, frame_end, fps = int(d.get("frame_start", 0)), d.get("frame_end"), d.get("fps")
        if d.get("truck"):
            from .truck_dataset import TruckDataset
            truck_cfg = dict(d["truck"])
            dataset = TruckDataset.discover(
                truck_cfg.get("root", "huniashahatona/zadania"),
                repairs_path=truck_cfg.get("repairs", "datasets/truck/ground_truth_repairs.json"),
            )
            lo, hi = dataset.common_frame_range()
            frame_start = int(truck_cfg.get("frame_start", lo))
            frame_end = int(truck_cfg.get("frame_end", hi))
            fps = float(truck_cfg.get("fps", dataset.fps))
            cameras = [dataset.calibration(name, frame_start) for name in dataset.names]
        elif d.get("sim"):
            from .sim import make_sim_setup
            sim = dict(d["sim"])
            cameras = make_sim_setup(**sim).cameras
            fps = float(sim.get("fps", fps or 25.0))
            frame_end = sim.get("num_frames", frame_end)
        elif d.get("calibration"):
            cameras, _ = load_calibration(d["calibration"])
        elif d.get("cameras"):
            cameras = [CameraCalibration.from_dict(c) for c in d["cameras"]]
        else:
            raise ValueError("конфиг: нужен один из truck / sim / calibration / cameras")
        streams = [StreamConfig.from_dict(s) for s in d.get("streams", [])]
        if dataset is not None and not streams:
            streams = [StreamConfig(kind="truck", name=n) for n in dataset.names]
        return cls(
            cameras=cameras,
            streams=streams,
            detector=d.get("detector", {"kind": "blob", "colors": []}),
            fusion=fusion,
            align_tolerance=float(d.get("align_tolerance", 0.05)),
            target_fps=d.get("target_fps"),
            stall_seconds=float(d.get("stall_seconds", 1.0)),
            sim=d.get("sim"),
            fps=fps,
            frame_start=frame_start,
            frame_end=None if frame_end is None else int(frame_end),
            truck=d.get("truck"),
            truck_dataset=dataset,
            partial_cameras=bool(d.get("partial_cameras", True)),
            min_cameras_live=int(d.get("min_cameras_live", 1)),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "PipelineConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class MultiViewPipeline:
    """Конвейер: синхронные детекции всех камер -> 3D-треки с историей."""

    def __init__(self, cameras: list[CameraCalibration],
                 config: PipelineConfig | None = None, *,
                 camera_provider: Any = None, fps: float | None = None) -> None:
        self.cameras = list(cameras)
        self.names = [c.name for c in cameras]
        self.config = config or PipelineConfig(cameras=cameras)
        self.tracker = MultiViewTracker(self.names, self.cameras, self.config.fusion)
        self.provider = camera_provider
        self.fps = fps if fps is not None else self.config.fps
        # track_id -> {observations, points, states, covariances, pos_var, vel_var}
        self.track_history: dict[int, dict[str, Any]] = {}
        self.step_times: list[float] = []
        self.stage_times: dict[str, list[float]] = {
            "read_detect": [], "fusion": [], "frame": [],
        }
        self.frame_times: list[float] = []
        self.frame_indices: list[int] = []
        self.last_result: FrameFusionResult | None = None
        # coverage по статусам (§7.6): сколько кадров дали измерение/прогноз/ничего
        self.coverage: dict[str, int] = {
            "frames": 0, "measured": 0, "predicted_only": 0, "none": 0,
            "monocular": 0, "views_histogram": {},
        }

    # ---------- сборка источников из конфига ----------

    def build_sources(self, *, repeat: bool | None = None) -> tuple[list[FrameSource], Any]:
        """Источники кадров по ``config.streams``; второй элемент — синтетика или None.

        ``kind=file``/``stream`` оборачиваются в ``DetectingFrameSource`` —
        детектор реально запускается на кадрах файла (дефект P0.3 брифа).
        ``kind=truck`` берёт видео из ``TruckDataset`` с номером кадра из CSV.
        """
        dataset = None
        sim_dataset = None
        if self.config.sim:
            from .sim import make_sim_setup
            sim_dataset = make_sim_setup(**self.config.sim)
        sources: list[FrameSource] = []
        for sc in self.config.streams:
            spec = sc.detector or self.config.detector
            if sc.kind == "sim":
                if sim_dataset is None:
                    raise ValueError("stream kind=sim, но в конфиге нет секции sim")
                loop = sc.repeat if repeat is None else repeat
                if spec.get("kind") == "oracle":
                    from .sim import OracleSimSource
                    src: FrameSource = OracleSimSource(sim_dataset, sc.sim_camera_index,
                                                       repeat=loop)
                else:
                    from .sim import SimCameraStream, SimSource
                    src = SimSource(SimCameraStream(sim_dataset, sc.sim_camera_index,
                                                    repeat=loop),
                                    build_detector(spec))
                if sc.name:
                    src.name = sc.name
                sources.append(src)
            elif sc.kind == "truck":
                ds = self.config.truck_dataset
                if ds is None:
                    raise ValueError("stream kind=truck, но в конфиге нет секции truck")
                cam = next((c for c in ds.cameras if c.name == sc.name), None)
                if cam is None:
                    raise ValueError(f"камера {sc.name!r} не найдена в датасете")
                from .truck_dataset import TruckVideoSource
                raw = TruckVideoSource(cam,
                                       start=sc.frame_start if sc.frame_start is not None
                                       else self.config.frame_start,
                                       end=sc.frame_end if sc.frame_end is not None
                                       else self.config.frame_end,
                                       fps=self.fps or ds.fps)
                sources.append(DetectingFrameSource(raw, build_detector(spec), name=sc.name))
            elif sc.kind in ("file", "stream"):
                from .capture import OpenCvFrameSource
                raw = OpenCvFrameSource(sc.source)
                name = sc.name or f"cam{len(sources) + 1}"
                sources.append(DetectingFrameSource(raw, build_detector(spec), name=name))
            else:
                raise ValueError(f"неизвестный kind потока: {sc.kind!r}")
        return sources, sim_dataset if sim_dataset is not None else dataset

    # ---------- шаги ----------

    def frame_of(self, timestamp: float) -> int | None:
        """Номер кадра по таймкоду (``round(t * fps)``); без fps — ``None``.

        ``None`` означает «номер кадра неизвестен»: тогда ``TrackPoint.frame``
        остаётся пустым, а метрики сопоставляют кадр по ``timestamp * fps``
        самостоятельно (как и раньше для синтетики).
        """
        if not self.fps:
            return None
        return int(round(timestamp * self.fps))

    def step(self, timestamp: float,
             detections_by_name: dict[str, list[Detection2D]],
             frame: int | None = None) -> FrameFusionResult:
        """Один кадр: детекции по имени камеры -> слияние/трекинг; обновляет историю."""
        t_frame = time.perf_counter()
        idx = self.frame_of(timestamp) if frame is None else int(frame)
        if self.provider is not None:
            if idx is None:
                raise ValueError(
                    "для CameraStateProvider нужен fps (номер кадра): задайте config.fps "
                    "или передайте frame= в step()")
            cams = self.provider.calibrations_at(idx, nearest=True)
            ordered = [cams.get(n, self.tracker.cameras[i]) for i, n in enumerate(self.names)]
            self.tracker.set_cameras(ordered)
            self.cameras = ordered
        detections = [detections_by_name.get(name, []) for name in self.names]
        t0 = time.perf_counter()
        result = self.tracker.step(timestamp, detections, frame=idx)
        t_fusion = time.perf_counter() - t0
        self.step_times.append(t_fusion)
        self.stage_times["fusion"].append(t_fusion)
        self.frame_times.append(timestamp)
        self.frame_indices.append(idx)
        self._update_coverage(result)
        for track in result.tracks:
            self.track_history[track.track_id] = {
                "observations": list(track.observations),
                "points": list(track.points),
                "states": [s for _, s, _ in track.state_history],
                "covariances": [p for _, _, p in track.state_history],
                "pos_var": track.filter.pos_var,
                "vel_var": track.filter.vel_var,
            }
        self.last_result = result
        self.stage_times["frame"].append(time.perf_counter() - t_frame)
        return result

    def _update_coverage(self, result: FrameFusionResult) -> None:
        """Покрытие по фактическим трекам кадра (а не по «сырым» кандидатам).

        Кандидат, отброшенный гейтом ассоциации, не должен считаться кадром с
        измерением: иначе coverage завышается ложными срабатываниями.
        """
        cov = self.coverage
        cov["frames"] += 1
        measured = []
        predicted = 0
        for track in result.tracks:
            if not track.points:
                continue
            last = track.points[-1]
            if last.timestamp != result.timestamp:
                continue
            if last.status == MEASURED:
                measured.append(last)
            elif last.status == PREDICTED_ONLY:
                predicted += 1
        if measured:
            cov["measured"] += 1
            views = len(measured[0].views)
            hist = cov["views_histogram"]
            hist[str(views)] = hist.get(str(views), 0) + 1
            if views == 1:
                cov["monocular"] += 1
        elif predicted:
            cov["predicted_only"] += 1
        else:
            cov["none"] += 1

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
                    align_tolerance: float | None = None,
                    frame_start: int | None = None) -> list[Track]:
        """Покадровый офлайн-проход; возвращает треки до RTS-сглаживания.

        Время чтения+детекции и время слияния меряются раздельно (per-stage
        timings §7.6). ``frame_start`` — номер кадра, соответствующий первому
        прочитанному кадру (для датасета «Пеленга» это начало общей области).
        """
        tol = self.config.align_tolerance if align_tolerance is None else align_tolerance
        n = 0
        t_prev = time.perf_counter()
        for aligned in read_all_offline(sources, align_tolerance=tol,
                                        max_frames=max_frames):
            t_read = time.perf_counter() - t_prev
            timestamp = float(aligned[0][0])
            frame = (self.frame_of(timestamp) if frame_start is None
                     else int(frame_start) + n)
            self.stage_times["read_detect"].append(t_read)
            self.step(timestamp, self._detections_from_aligned(aligned), frame=frame)
            n += 1
            t_prev = time.perf_counter()
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
                aligned = read_aligned(
                    readers, align_tolerance=self.config.align_tolerance,
                    timeout=0.1,
                    min_cameras=self.config.min_cameras_live,
                    allow_partial=self.config.partial_cameras,
                )
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
        """Треки по измеренным точкам (отфильтрованным Калманом, до RTS)."""
        return [Track(tid, list(h["observations"]))
                for tid, h in sorted(self.track_history.items()) if h["observations"]]

    def all_tracks(self) -> list[Track]:
        """Треки, включая прогнозные точки (``PREDICTED_ONLY``) — для coverage/экспорта."""
        return [Track(tid, list(h["points"]))
                for tid, h in sorted(self.track_history.items()) if h["points"]]

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
                                 obs[i].views, obs[i].mean_reproj_px,
                                 frame=obs[i].frame, method=obs[i].method,
                                 status=obs[i].status, confidence=obs[i].confidence)
                      for i, s in enumerate(smoothed)]
            out.append(Track(tid, points))
        return out

    def stats(self, readers: list[FrameReader] | None = None) -> dict[str, Any]:
        """Частота обработки, время шага, per-stage тайминги, покрытие (для live-режима)."""
        n = len(self.step_times)
        total = sum(self.step_times)
        read = self.stage_times.get("read_detect", [])
        st: dict[str, Any] = {
            "frames": n,
            "processing_fps": (n / total) if total > 1e-9 else 0.0,
            "step_ms_mean": (1000.0 * total / n) if n else 0.0,
            "step_ms_max": (1000.0 * max(self.step_times)) if n else 0.0,
            "read_detect_ms_mean": (1000.0 * sum(read) / len(read)) if read else 0.0,
            "end_to_end_ms_mean": (
                1000.0 * (total + sum(read)) / n) if n else 0.0,
            "coverage": dict(self.coverage),
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
