"""Адаптеры источников кадров: детектор поверх любого ``FrameSource``.

Дефект P0.3 брифа: ``MultiViewPipeline.build_sources`` создавал
``OpenCvFrameSource`` без детектора, поэтому на файлах/RTSP детекции никогда не
вычислялись. Здесь добавлен декоратор ``DetectingFrameSource``: он оборачивает
любой источник кадров и прогоняет кадр через ``Detector``, возвращая
``(таймкод, кадр, [Detection2D, ...])`` — ровно тот контракт, который ждёт
``capture.FrameReader`` и конвейер.

Детектор один на источник (per-stream detector из конфига) либо общий
(``global detector``): ``build_sources`` передаёт спецификацию явно.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .types import Detection2D


class DetectingFrameSource:
    """Источник кадров с детекцией: обёртка над ``FrameSource`` + ``Detector``.

    Parameters
    ----------
    source:
        Любой объект с ``read() -> (stamp, frame, dets|None)`` и ``close()``.
    detector:
        Объект с ``detect(frame_bgr) -> list[Detection2D]``.
    name:
        Имя камеры; подставляется в ``Detection2D.camera``. Если не задано,
        берётся ``source.name``.
    clip_to_frame:
        Отбрасывать детекции, центр которых вне кадра (полезно для моделей,
        отдающих боксы за границей изображения).
    """

    def __init__(self, source, detector, *, name: str = "",
                 clip_to_frame: bool = True) -> None:
        self.source = source
        self.detector = detector
        self.name = name or getattr(source, "name", "") or "cam"
        self.clip_to_frame = clip_to_frame
        self.frames_detected = 0
        self.detections_total = 0

    # ---------- протокол FrameSource ----------

    def read(self) -> tuple[float, np.ndarray, list[Detection2D]] | None:
        item = self.source.read()
        if item is None:
            return None
        stamp, frame, dets = item
        if dets is None:
            dets = self.detector.detect(frame)
        else:
            dets = list(dets)
        dets = self._normalize(dets, frame, stamp)
        self.frames_detected += 1
        self.detections_total += len(dets)
        return stamp, frame, dets

    def _normalize(self, dets: Iterable[Detection2D], frame: np.ndarray,
                   stamp: float) -> list[Detection2D]:
        h, w = frame.shape[:2]
        out: list[Detection2D] = []
        for det in dets:
            det.camera = self.name
            det.timestamp = float(stamp)
            if self.clip_to_frame:
                u, v = float(det.center[0]), float(det.center[1])
                if not (0.0 <= u < w and 0.0 <= v < h):
                    continue
            out.append(det)
        return out

    @property
    def closed(self) -> bool:
        return bool(getattr(self.source, "closed", False))

    @property
    def frame_index(self) -> int | None:
        """Номер текущего кадра, если источник его знает (для dynamic cameras)."""
        return getattr(self.source, "frame_index", None)

    def close(self) -> None:
        close = getattr(self.source, "close", None)
        if close is not None:
            close()

    def __repr__(self) -> str:  # pragma: no cover - диагностика
        return f"DetectingFrameSource(name={self.name!r}, source={self.source!r})"


def wrap_with_detector(sources: Iterable, detector, names: Iterable[str] | None = None
                       ) -> list[DetectingFrameSource]:
    """Оборачивает последовательность источников одним общим детектором."""
    names = list(names) if names is not None else ["" for _ in sources]
    return [DetectingFrameSource(src, detector, name=name)
            for src, name in zip(sources, names)]
