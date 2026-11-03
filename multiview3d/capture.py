"""Захват кадров с камер: файлы, RTSP/USB, синтетика; выравнивание по времени.

Задержка и потерянные кадры компенсируются так:
- каждый читатель держит только **последний** кадр (старые не буферизуются,
  поэтому очередь не растёт при просадках);
- кадры разных камер на одном шаге конвейера берутся по ближайшему таймкоду
  с допуском ``align_tolerance``; кадры старше допуска считаются потерянными;
- целевая частота(converge_to_fps) ограничивает темп, чтобы детектор и
  триангуляция не работали быстрее, чем приходят камеры.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Protocol, Sequence

import cv2
import numpy as np

from .types import Detection2D


class FrameSource(Protocol):
    """Источник кадров: ``read() -> (таймкод, кадр BGR, детекции|None)`` или None."""

    closed: bool

    def read(self) -> tuple[float, np.ndarray, list[Detection2D] | None] | None: ...

    def close(self) -> None: ...


@dataclass
class ReaderStats:
    """Метрики задержки и потерь для одной камеры."""

    name: str
    frames_read: int = 0
    frames_dropped: int = 0
    last_latency_ms: float = 0.0
    mean_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    fps_source: float = 0.0
    _lat_sum: float = field(default=0.0, repr=False)

    def update(self, latency_ms: float) -> None:
        self.frames_read += 1
        self._lat_sum += latency_ms
        self.mean_latency_ms = self._lat_sum / self.frames_read
        self.max_latency_ms = max(self.max_latency_ms, latency_ms)
        self.last_latency_ms = latency_ms


class LatestFrameBuffer:
    """Буфер «только последний кадр»: новые вытесняют старые.

    ``get()`` отдаёт **самый свежий** элемент (политика latest, а не FIFO):
    при просадке обработчика очередь не накапливает устаревшие кадры —
    иначе конвейер систематически работал бы по задержанным данным.
    """

    def __init__(self, maxlen: int = 2) -> None:
        self._q: queue.Queue = queue.Queue(maxsize=maxlen)
        self._lock = threading.Lock()

    def put(self, item) -> None:
        with self._lock:
            if self._q.full():
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass
            self._q.put_nowait(item)

    def get(self, timeout: float | None = None):
        """Самый новый элемент или None по таймауту/пустому буферу."""
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        while True:
            with self._lock:
                item = None
                while True:                      # вычерпываем очередь до последнего
                    try:
                        item = self._q.get_nowait()
                    except queue.Empty:
                        break
                if item is not None:
                    return item
            if deadline is None:
                return None
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            time.sleep(min(0.005, remaining))


class OpenCvFrameSource:
    """Файл видео, изображение или RTSP/USB-поток через cv2.VideoCapture."""

    def __init__(self, source: str | int, *, size: tuple[int, int] | None = None,
                 fps_fallback: float = 30.0) -> None:
        self.source = source
        self.size = size
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"не удалось открыть источник: {source}")
        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS) or fps_fallback)
        self._index = 0

    @property
    def closed(self) -> bool:
        return not self._cap.isOpened()

    def read(self) -> tuple[float, np.ndarray, None] | None:
        ok, frame = self._cap.read()
        if not ok:
            return None
        if self.size is not None and (frame.shape[1], frame.shape[0]) != tuple(self.size):
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        stamp = self._index / self.fps
        self._index += 1
        return stamp, frame, None

    def close(self) -> None:
        self._cap.release()


class FrameReader:
    """Фоновый читатель одного источника; хранит последний кадр и статистику."""

    def __init__(self, source: FrameSource, name: str = "") -> None:
        self.source = source
        self.name = name or getattr(source, "name", None) or str(source.source)
        self.stats = ReaderStats(name=self.name)
        self._buf = LatestFrameBuffer(maxlen=2)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name=f"reader-{self.name}", daemon=True)

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            t0 = time.perf_counter()
            try:
                item = self.source.read()
            except Exception:
                item = None
            if item is None:
                self.stats.frames_dropped += 1
                self._stop.wait(0.02)
                continue
            stamp, frame, detections = item
            latency_ms = (time.perf_counter() - t0) * 1000.0
            self.stats.update(latency_ms)
            self.stats.fps_source = 1.0 / max(1e-6, time.perf_counter() - t0)
            self._buf.put((stamp, frame, detections, time.monotonic()))

    def latest(self, *, timeout: float = 0.0):
        """Последний кадр ``(таймкод, кадр, детекции, monotonic)`` или None.

        ``timeout <= 0`` — неблокирующий опрос буфера.
        """
        if timeout <= 0:
            try:
                return self._buf.get(timeout=0)
            except queue.Empty:
                return None
        return self._buf.get(timeout=timeout)

    @property
    def closed(self) -> bool:
        return self.source.closed

    def close(self) -> None:
        self._stop.set()
        self.source.close()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    @staticmethod
    def create(source: str | int, *, size: tuple[int, int] | None = None) -> "FrameReader":
        return FrameReader(OpenCvFrameSource(source, size=size), name=str(source))


def build_readers(sources: Sequence[FrameSource], names: Sequence[str] | None = None) -> list[FrameReader]:
    """Создаёт и запускает читатели по списку источников (имена — ``src.name`` или ``cam{i}``)."""
    readers = []
    for i, src in enumerate(sources):
        name = names[i] if names is not None else (getattr(src, "name", None) or f"cam{i}")
        readers.append(FrameReader(src, name=str(name)))
    for r in readers:
        r.start()
    return readers


def read_aligned(readers: Sequence[FrameReader], *, align_tolerance: float = 0.05,
                 timeout: float = 0.5, min_cameras: int | None = None,
                 allow_partial: bool = False) -> dict[str, tuple] | None:
    """Собирает по одному последнему кадру с камер в пределах допуска.

    Возвращает ``{имя: (таймкод, кадр, детекции, monotonic)}`` (4-tuple из
    ``FrameReader.latest``) или None.

    ``allow_partial=False`` (историческое поведение) — нужен полный набор
    камер, иначе None. ``allow_partial=True`` — достаточно ``min_cameras``
    (по умолчанию 1): кадры камер, отставших сильнее ``align_tolerance``,
    отбрасываются, остальные идут в результат. Так поддерживается частичная
    доступность камер (P1.1 брифа): объект может быть виден не всеми видами.
    """
    frames: dict[str, tuple] = {}
    best_stamp = None
    for r in readers:
        item = r.latest(timeout=timeout)
        if item is None:
            continue
        frames[r.name] = item
        best_stamp = item[0] if best_stamp is None else max(best_stamp, item[0])
    if best_stamp is None:
        return None
    aligned = {k: v for k, v in frames.items() if abs(v[0] - best_stamp) <= align_tolerance}
    need = len(readers) if not allow_partial else int(min_cameras or 1)
    if len(aligned) < need:
        for r in readers:
            if r.name not in aligned:
                r.stats.frames_dropped += 1
        return None
    return aligned


def read_all_offline(sources: Sequence[FrameSource], *, align_tolerance: float = 0.05,
                     max_frames: int | None = None, max_skip: int = 1000):
    """Офлайн-проход: генератор выровненных кадров до исчерпания источников.

    Каждый шаг читает по одному кадру от каждого источника и выдаёт словарь
    ``{i: (таймкод, кадр, детекции)}``; при расхождении таймкодов сильнее
    допуска кадры пропускаются (счётчик ``max_skip`` защищает от бесконечного
    пропуска). Завершается, когда любой источник вернул None или выдано
    ``max_frames`` кадров.
    """
    emitted = 0
    skipped = 0
    while max_frames is None or emitted < max_frames:
        items = []
        for src in sources:
            item = src.read()
            if item is None:
                return
            items.append(item)
        stamps = [it[0] for it in items]
        if max(stamps) - min(stamps) > align_tolerance:
            skipped += 1
            if skipped > max_skip:
                return
            continue
        skipped = 0
        yield {i: it for i, it in enumerate(items)}
        emitted += 1


def read_first_aligned(sources: Sequence[FrameSource], *, align_tolerance: float = 0.05):
    """Первый выровненный кадр офлайн-прохода или None (источники исчерпаны)."""
    for frame in read_all_offline(sources, align_tolerance=align_tolerance):
        return frame
    return None
