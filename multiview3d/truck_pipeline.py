"""Стадии truck-workflow: детекция по пяти видео, реконструкция, экспорт.

Модуль отделяет «инженерные» детали от CLI-приложений:

* ``detect_frames`` — детектор по кадрам всех камер набора (последовательное
  чтение, таймкод = frame / fps);
* ``save_detections`` / ``load_detections`` — CSV-обмен 2D-наблюдениями между
  этапами (одна строка = одна детекция);
* ``run_reconstruction`` — покадровое слияние с позами ``CameraStateProvider``
  (cam5 вращается) и калиброванным сдвигом якоря;
* ``export_estimates`` — CSV/JSON оценок 3D в DTO §11 брифа
  (frame/time/method/status/camera_ids/confidence/uncertainty).

Все стадии не изменяют исходные файлы датасета.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .capture import FrameSource
from .fusion import FusionConfig, MultiViewTracker, canonical_view_key
from .pipeline import MultiViewPipeline, PipelineConfig
from .truck_dataset import TruckDataset, TruckVideoSource
from .types import (
    Detection2D,
    MEASURED,
    METHOD_TRIANGULATION,
    PREDICTED_ONLY,
    TrackPoint,
)

DETECTION_COLUMNS = ["camera", "frame", "timestamp", "score", "label",
                     "x1", "y1", "x2", "y2", "cx", "cy"]
ESTIMATE_COLUMNS = ["frame", "timestamp", "x", "y", "z", "vx", "vy", "vz",
                    "status", "method", "n_views", "views", "mean_reproj_px",
                    "confidence"]


# --------------------------------------------------------------------------
# 2D-детекции: обмен между этапами
# --------------------------------------------------------------------------


def detection_to_row(det: Detection2D, frame: int) -> dict[str, Any]:
    """Детекция -> строка CSV (bbox может отсутствовать: тогда пустые поля)."""
    x1, y1, x2, y2 = det.bbox if det.bbox is not None else ("", "", "", "")
    return {
        "camera": det.camera,
        "frame": int(frame),
        "timestamp": round(float(det.timestamp), 6),
        "score": round(float(det.score), 5),
        "label": det.label,
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "cx": round(float(det.center[0]), 3),
        "cy": round(float(det.center[1]), 3),
    }


def save_detections(rows: Iterable[dict[str, Any]], path: str | Path) -> int:
    """Запись 2D-детекций в CSV; возвращает число строк."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=DETECTION_COLUMNS)
        w.writeheader()
        for row in rows:
            w.writerow(row)
            n += 1
    return n


def load_detections(path: str | Path) -> dict[int, dict[str, list[Detection2D]]]:
    """Чтение CSV детекций -> ``{frame: {camera: [Detection2D, ...]}}``."""
    out: dict[int, dict[str, list[Detection2D]]] = {}
    with Path(path).open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            frame = int(row["frame"])
            cam = row["camera"]
            bbox = None
            if row.get("x1") not in ("", None):
                bbox = (float(row["x1"]), float(row["y1"]),
                        float(row["x2"]), float(row["y2"]))
            det = Detection2D(
                camera=cam,
                timestamp=float(row["timestamp"]),
                center=np.array([float(row["cx"]), float(row["cy"])]),
                score=float(row["score"]),
                bbox=bbox,
                label=row.get("label", ""),
            )
            out.setdefault(frame, {}).setdefault(cam, []).append(det)
    return out


# --------------------------------------------------------------------------
# Стадия 1: детекция
# --------------------------------------------------------------------------


@dataclass
class DetectStats:
    frames: int = 0
    frames_with_detection: int = 0
    detections: int = 0
    per_camera: dict[str, int] | None = None
    seconds: float = 0.0
    seconds_per_frame: float = 0.0
    fps_processing: float = 0.0


def detect_frames(dataset: TruckDataset, detector, frames: Sequence[int], *,
                  progress: Callable[[int, int], None] | None = None,
                  log_every: int = 0) -> tuple[dict[int, dict[str, list[Detection2D]]], DetectStats]:
    """Прогоняет детектор по кадрам ``frames`` всех камер набора.

    Чтение последовательное (``TruckVideoSource``): кадры между запрошенными
    декодируются, но не отдаются детектору. Таймкод каждого кадра —
    ``frame / fps``. Порядок кадров — возрастающий.
    """
    frames = sorted(int(f) for f in frames)
    if not frames:
        return {}, DetectStats()
    sources: dict[str, TruckVideoSource] = {
        cam.name: TruckVideoSource(cam, start=frames[0], end=frames[-1], fps=dataset.fps)
        for cam in dataset.cameras
    }
    out: dict[int, dict[str, list[Detection2D]]] = {}
    stats = DetectStats(per_camera={n: 0 for n in sources})
    t0 = time.perf_counter()
    try:
        for i, frame in enumerate(frames):
            per_cam: dict[str, list[Detection2D]] = {}
            for name, src in sources.items():
                while src.frame_index < frame:
                    if src.read() is None:
                        raise RuntimeError(
                            f"{name}: не удалось дочитать до кадра {frame}")
                item = src.read()
                if item is None:
                    raise RuntimeError(f"{name}: кадр {frame} не читается")
                stamp, image, _ = item
                dets = list(detector.detect(image))
                for det in dets:
                    det.camera = name
                    det.timestamp = float(stamp)
                per_cam[name] = dets
                stats.per_camera[name] = stats.per_camera.get(name, 0) + len(dets)
                stats.detections += len(dets)
            out[frame] = per_cam
            stats.frames += 1
            if any(per_cam.values()):
                stats.frames_with_detection += 1
            if log_every and (i + 1) % log_every == 0:
                print(f"  обработано кадров {i + 1}/{len(frames)} "
                      f"({time.perf_counter() - t0:.1f} с)")
            if progress is not None:
                progress(i + 1, len(frames))
    finally:
        for src in sources.values():
            src.close()
    stats.seconds = time.perf_counter() - t0
    stats.seconds_per_frame = stats.seconds / max(stats.frames, 1)
    stats.fps_processing = stats.frames / max(stats.seconds, 1e-9)
    return out, stats


def detection_rows(detections: dict[int, dict[str, list[Detection2D]]]
                   ) -> Iterable[dict[str, Any]]:
    for frame in sorted(detections):
        for cam in sorted(detections[frame]):
            for det in detections[frame][cam]:
                yield detection_to_row(det, frame)


# --------------------------------------------------------------------------
# Стадия 2: реконструкция
# --------------------------------------------------------------------------


def build_pipeline(dataset: TruckDataset, *, fusion: FusionConfig | None = None,
                   anchor_offset_m: Sequence[float] | None = None,
                   anchor_px_offset: dict[str, Sequence[float]] | None = None,
                   anchor_offset_views: dict[str, Sequence[float]] | None = None
                   ) -> MultiViewPipeline:
    """Конвейер с провайдером поз по кадру и (опционально) сдвигами якоря."""
    cfg = fusion or FusionConfig()
    if anchor_offset_m is not None:
        cfg.anchor_offset_m = tuple(float(v) for v in anchor_offset_m)
    if anchor_px_offset is not None:
        cfg.anchor_px_offset = {k: (float(v[0]), float(v[1]))
                                for k, v in anchor_px_offset.items()}
    if anchor_offset_views is not None:
        cfg.anchor_offset_views = {
            canonical_view_key(k.split("|")): tuple(float(v) for v in vec)
            for k, vec in anchor_offset_views.items()}
    frame0 = dataset.common_frame_range()[0]
    cameras = [dataset.calibration(name, frame0) for name in dataset.names]
    pipe_cfg = PipelineConfig(cameras=cameras, fusion=cfg, fps=dataset.fps,
                              frame_start=frame0)
    return MultiViewPipeline(cameras, config=pipe_cfg,
                             camera_provider=dataset.provider, fps=dataset.fps)


@dataclass
class ReconstructionResult:
    pipeline: MultiViewPipeline
    frames: list[int]
    seconds: float = 0.0
    seconds_per_frame: float = 0.0


def run_reconstruction(dataset: TruckDataset, detections: dict[int, dict[str, list[Detection2D]]],
                       *, fusion: FusionConfig | None = None,
                       anchor_offset_m: Sequence[float] | None = None,
                       anchor_px_offset: dict[str, Sequence[float]] | None = None,
                       anchor_offset_views: dict[str, Sequence[float]] | None = None,
                       frames: Sequence[int] | None = None,
                       log_every: int = 0) -> ReconstructionResult:
    """Покадровое слияние: обновляет позы камер, строит 3D-треки и статусы."""
    pipe = build_pipeline(dataset, fusion=fusion, anchor_offset_m=anchor_offset_m,
                          anchor_px_offset=anchor_px_offset,
                          anchor_offset_views=anchor_offset_views)
    order = sorted(int(f) for f in (frames if frames is not None else detections))
    t0 = time.perf_counter()
    for i, frame in enumerate(order):
        per_cam = detections.get(frame, {})
        pipe.step(frame / dataset.fps, per_cam, frame=frame)
        if log_every and (i + 1) % log_every == 0:
            print(f"  реконструкция {i + 1}/{len(order)}")
    dt = time.perf_counter() - t0
    return ReconstructionResult(pipeline=pipe, frames=order, seconds=dt,
                                seconds_per_frame=dt / max(len(order), 1))


def estimate_points(result: ReconstructionResult) -> list[TrackPoint]:
    """Все точки (измерения + прогнозы) в хронологическом порядке по кадрам."""
    pts: list[TrackPoint] = []
    for track in result.pipeline.all_tracks():
        pts.extend(track.points)
    return pts


def load_estimates(path: str | Path) -> list[TrackPoint]:
    """Чтение CSV оценок (обратная операция к :func:`export_estimates`)."""
    pts: list[TrackPoint] = []
    with Path(path).open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            views = [v for v in (row.get("views") or "").split("|") if v]
            num = lambda key: (float(row[key]) if row.get(key) not in (None, "")  # noqa: E731
                               else float("nan"))
            pts.append(TrackPoint(
                timestamp=float(row["timestamp"]),
                position=np.array([float(row["x"]), float(row["y"]), float(row["z"])]),
                velocity=np.array([num("vx"), num("vy"), num("vz")]),
                views=views,
                mean_reproj_px=num("mean_reproj_px"),
                frame=int(row["frame"]) if row.get("frame") not in (None, "") else None,
                method=row.get("method") or METHOD_TRIANGULATION,
                status=row.get("status") or MEASURED,
                confidence=num("confidence"),
            ))
    return pts


def export_estimates(result: ReconstructionResult, path: str | Path) -> int:
    """Экспорт оценок в CSV (колонки DTO §11); возвращает число строк."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    by_frame: dict[int, TrackPoint] = {}
    for pt in estimate_points(result):
        f = pt.frame if pt.frame is not None else int(round(pt.timestamp * result.pipeline.fps))
        cur = by_frame.get(f)
        if cur is None or (pt.status == MEASURED and cur.status != MEASURED):
            by_frame[f] = pt
    n = 0
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(ESTIMATE_COLUMNS)
        for f in sorted(by_frame):
            pt = by_frame[f]
            w.writerow([
                f, f"{pt.timestamp:.6f}",
                *[f"{v:.6f}" for v in np.asarray(pt.position, dtype=float)],
                *[f"{v:.6f}" for v in np.asarray(pt.velocity, dtype=float)],
                pt.status, pt.method, len(pt.views), "|".join(pt.views),
                (f"{pt.mean_reproj_px:.3f}" if np.isfinite(pt.mean_reproj_px) else ""),
                (f"{pt.confidence:.4f}" if np.isfinite(pt.confidence) else ""),
            ])
            n += 1
    return n


def estimates_json(result: ReconstructionResult, *, meta: dict[str, Any] | None = None
                   ) -> dict[str, Any]:
    """JSON: ``{"meta": ..., "estimates": [DTO §11], "coverage": ...}``."""
    rows = []
    for track in result.pipeline.all_tracks():
        for pt in sorted(track.points, key=lambda q: (q.timestamp, q.status)):
            rows.append({
                "track_id": track.track_id,
                "frame_id": pt.frame,
                "timestamp": float(pt.timestamp),
                "position_xyz_m": [float(v) for v in pt.position],
                "velocity_xyz_m_s": [float(v) for v in pt.velocity],
                "method": pt.method,
                "status": pt.status,
                "camera_ids": list(pt.views),
                "reprojection_error_px": (float(pt.mean_reproj_px)
                                          if np.isfinite(pt.mean_reproj_px) else None),
                "confidence": (float(pt.confidence)
                               if np.isfinite(pt.confidence) else None),
            })
    return {
        "meta": meta or {},
        "coverage": dict(result.pipeline.coverage),
        "stats": _stats_json(result),
        "estimates": rows,
    }


def _stats_json(result: ReconstructionResult) -> dict[str, Any]:
    n = len(result.frames)
    sec = result.seconds
    return {
        "frames": n,
        "seconds": round(sec, 3),
        "fps_processing": round(n / sec, 2) if sec > 1e-9 else None,
        "ms_per_frame": round(1000.0 * sec / n, 2) if n else None,
    }


def write_estimates_json(result: ReconstructionResult, path: str | Path,
                         *, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    doc = estimates_json(result, meta=meta)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    return doc


# --------------------------------------------------------------------------
# Вспомогательное
# --------------------------------------------------------------------------


def frames_range(dataset: TruckDataset, *, start: int | None = None,
                 end: int | None = None, step: int = 1) -> list[int]:
    """Список кадров ``start..end`` с шагом ``step`` в пределах общей области."""
    lo, hi = dataset.common_frame_range()
    start = lo if start is None else int(start)
    end = hi if end is None else int(end)
    step = max(1, int(step))
    start = max(start, lo)
    end = min(end, hi)
    return list(range(start, end + 1, step))


def load_anchor_offset(path: str | Path) -> list[float] | None:
    """Считывает ``{"offset_m": [dx, dy, dz]}`` из JSON (или None, если файла нет)."""
    p = Path(path)
    if not p.exists():
        return None
    doc = json.loads(p.read_text(encoding="utf-8-sig"))
    offset = doc.get("offset_m") or doc.get("dx_m")
    if offset is None:
        return None
    return [float(v) for v in offset]


def load_anchor_px_offset(path: str | Path) -> dict[str, list[float]] | None:
    """Считывает ``{"offsets_px": {"cam1": [du, dv], ...}}`` (или None)."""
    p = Path(path)
    if not p.exists():
        return None
    doc = json.loads(p.read_text(encoding="utf-8-sig"))
    offsets = doc.get("offsets_px") or doc.get("offsets")
    if not offsets:
        return None
    return {str(k): [float(v[0]), float(v[1])] for k, v in offsets.items()}


def load_anchor_offset_views(path: str | Path) -> dict[str, list[float]] | None:
    """Считывает ``{"offsets_m": {"cam3|cam5": [dx, dy, dz], ...}}`` (или None)."""
    p = Path(path)
    if not p.exists():
        return None
    doc = json.loads(p.read_text(encoding="utf-8-sig"))
    offsets = doc.get("offsets_m") or doc.get("offsets")
    if not offsets:
        return None
    return {canonical_view_key(str(k).split("|")): [float(v[0]), float(v[1]), float(v[2])]
            for k, v in offsets.items()}


def source_names(sources: Sequence[FrameSource]) -> list[str]:
    return [getattr(s, "name", f"cam{i}") for i, s in enumerate(sources)]
