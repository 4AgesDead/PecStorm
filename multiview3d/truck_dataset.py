"""Датасет «Пеленга»: пять MP4, пять camera CSV, GT-траектория.

Модуль изолирует входной формат организаторов (Unreal Engine export) от
общей геометрии ``multiview3d``:

* ``discover()`` находит файлы по шаблону и читает свойства видео (OpenCV);
* ``manifest()`` фиксирует размеры, fps, диапазоны кадров, sha256 исходников;
* ``common_frame_range()`` даёт **общую** область кадров без off-by-one:
  видео содержит кадры ``0..999`` при ``CAP_PROP_FRAME_COUNT == 1000``, а CSV
  камер — строки ``0..1001``; обработке подлежит пересечение;
* ``TruckVideoSource`` — последовательный читатель MP4 с таймкодом
  ``frame / fps`` (кадр — та же величина, что ``Frame`` в CSV, поэтому позы
  камеры и детекции синхронизированы по номеру кадра, а не по догадке);
* ``TruckDataset.provider`` — ``CameraStateProvider`` с позами по кадру
  (cam1–4 статичны, cam5 вращается).

Оригинальные файлы не изменяются: только чтение.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .camera_provider import CameraSpec, CameraStateProvider
from .unreal_camera import GroundTruth, load_ground_truth

DEFAULT_ROOT = Path("huniashahatona/zadania")
CAMERA_VIDEO_GLOB = "camera/Way1.FinalImage.CineCameraActor{index}.mp4"
CAMERA_CSV_GLOB = "xyzpowcameta/object_transform_CineCameraActor{index}.csv"
GT_CSV = "xyzpowcameta/object_transform_SM_AA_track.csv"
DEFAULT_REPAIRS = Path("datasets/truck/ground_truth_repairs.json")


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """SHA-256 файла (потоковое чтение, без загрузки целиком в память)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


@dataclass
class TruckCamera:
    """Одна камера набора: номер, видео, CSV параметров."""

    name: str
    index: int
    video: Path
    csv: Path


def _video_props(path: str | Path) -> dict:
    """Свойства видеофайла по данным контейнера (OpenCV, без полного декодирования)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"не удалось открыть видео: {path}")
    props = {
        "width": int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))),
        "height": int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        "frame_count": int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT))),
    }
    cap.release()
    return props


@dataclass
class TruckDataset:
    """Набор «Пеленга»: пути, свойства видео, провайдер поз, ground truth."""

    root: Path
    cameras: list[TruckCamera]
    gt_csv: Path
    repairs_path: Path | None = DEFAULT_REPAIRS
    fps: float = 30.0
    _props: dict[str, dict] = field(default_factory=dict, repr=False)
    _provider: CameraStateProvider | None = field(default=None, repr=False)
    _gt: GroundTruth | None = field(default=None, repr=False)

    # ---------- обнаружение файлов ----------

    @classmethod
    def discover(cls, root: str | Path = DEFAULT_ROOT,
                 repairs_path: str | Path | None = DEFAULT_REPAIRS) -> "TruckDataset":
        """Находит пять видео и пять CSV; падает с понятной ошибкой, если чего-то нет."""
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f"корень датасета не найден: {root}")
        cameras: list[TruckCamera] = []
        for i in range(1, 6):
            video = root / CAMERA_VIDEO_GLOB.format(index=i)
            csv_path = root / CAMERA_CSV_GLOB.format(index=i)
            for p in (video, csv_path):
                if not p.exists():
                    raise FileNotFoundError(f"нет файла датасета: {p}")
            cameras.append(TruckCamera(name=f"cam{i}", index=i, video=video, csv=csv_path))
        gt_csv = root / GT_CSV
        if not gt_csv.exists():
            raise FileNotFoundError(f"нет ground truth: {gt_csv}")
        ds = cls(root=root, cameras=cameras, gt_csv=gt_csv,
                 repairs_path=Path(repairs_path) if repairs_path else None)
        props = {cam.name: _video_props(cam.video) for cam in cameras}
        ds._props = props
        fps_set = {round(p["fps"], 3) for p in props.values()}
        sizes = {(p["width"], p["height"]) for p in props.values()}
        if len(fps_set) != 1 or len(sizes) != 1:
            raise ValueError(
                f"камеры с разной частотой/размером: fps={fps_set}, size={sizes}; "
                "выравнивание по frame-index требует одинаковых параметров"
            )
        ds.fps = float(next(iter(fps_set)))
        return ds

    # ---------- свойства и диапазоны ----------

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.cameras]

    @property
    def width(self) -> int:
        return int(self._props[self.names[0]]["width"])

    @property
    def height(self) -> int:
        return int(self._props[self.names[0]]["height"])

    def video_frame_count(self, name: str) -> int:
        """Число кадров, которые можно прочитать: индексы ``0..count-1``."""
        return int(self._props[name]["frame_count"])

    @property
    def provider(self) -> CameraStateProvider:
        """Провайдер поз (кэшируется); на статичных cam1–4 поза одна на все кадры."""
        if self._provider is None:
            specs = [CameraSpec(name=c.name, csv_path=str(c.csv),
                                width=self.width, height=self.height)
                     for c in self.cameras]
            self._provider = CameraStateProvider(specs)
        return self._provider

    def csv_frames(self, name: str) -> list[int]:
        return self.provider.frames_available(name)

    def common_frame_range(self) -> tuple[int, int]:
        """Общая область кадров ``(start, end)`` включительно.

        Берётся пересечение «читаемые кадры видео» и «строки CSV камер» по всем
        камерам. Для текущего набора это ``(0, 1000)``: видео 1001 кадр
        (индексы 0..1000), CSV камер — строки 0..1001 (лишняя строка 1001 не
        имеет видеокадра и в обработку не попадает).
        """
        starts, ends = [], []
        for cam in self.cameras:
            frames = self.csv_frames(cam.name)
            if not frames:
                raise ValueError(f"в CSV камеры {cam.name} нет строк с кадрами")
            starts.append(max(0, frames[0]))
            ends.append(min(self.video_frame_count(cam.name) - 1, frames[-1]))
        start, end = max(starts), min(ends)
        if end < start:
            raise ValueError(f"нет общей области кадров: start={start}, end={end}")
        return start, end

    def frame_count(self) -> int:
        start, end = self.common_frame_range()
        return end - start + 1

    def mismatch_report(self) -> dict:
        """Отчёт о расхождении числа кадров видео и строк CSV (§7.1 п.9)."""
        start, end = self.common_frame_range()
        per_camera = {}
        for cam in self.cameras:
            frames = self.csv_frames(cam.name)
            per_camera[cam.name] = {
                "video_frames": self.video_frame_count(cam.name),
                "video_last_index": self.video_frame_count(cam.name) - 1,
                "csv_frames": len(frames),
                "csv_first": frames[0] if frames else None,
                "csv_last": frames[-1] if frames else None,
                "csv_rows_without_video": max(
                    0, (frames[-1] + 1) - self.video_frame_count(cam.name)) if frames else 0,
            }
        gt_frames = self.ground_truth().frames
        return {
            "common_range": [start, end],
            "common_frames": end - start + 1,
            "per_camera": per_camera,
            "gt_first": int(gt_frames.min()) if gt_frames.size else None,
            "gt_last": int(gt_frames.max()) if gt_frames.size else None,
            "gt_frames": int(gt_frames.size),
        }

    # ---------- позы и GT ----------

    def calibration(self, name: str, frame: int):
        return self.provider.calibration(name, frame)

    def calibrations_at(self, frame: int, *, nearest: bool = False):
        return self.provider.calibrations_at(frame, nearest=nearest)

    def ground_truth(self, *, accept_unverified: bool | None = None,
                     repairs_path: str | Path | None = None) -> GroundTruth:
        """GT цели в метрах; ``accept_unverified`` перекрывает флаг sidecar-файла.

        По умолчанию репары не применяются (``accept_unverified`` из файла =
        false), поэтому кадры 402 и 691 остаются в ``invalid_cells`` и в GT
        отсутствуют — оценка идёт только по валидным кадрам, а факт исключения
        виден в отчёте.
        """
        rp = repairs_path if repairs_path is not None else self.repairs_path
        if accept_unverified is not None:
            return load_ground_truth(self.gt_csv, repairs_path=rp, strict=False,
                                     accept_unverified=accept_unverified)
        if self._gt is None:
            self._gt = load_ground_truth(self.gt_csv, repairs_path=rp, strict=False)
        return self._gt

    def ground_truth_by_frame(self, **kwargs) -> dict[int, np.ndarray]:
        gt = self.ground_truth(**kwargs)
        return {int(f): np.asarray(p, dtype=float) for f, p in zip(gt.frames, gt.positions_m)}

    # ---------- источники кадров ----------

    def open_sources(self, *, start: int | None = None, end: int | None = None,
                     names: list[str] | None = None) -> list["TruckVideoSource"]:
        """Последовательные читатели видео; ``end`` включительно, по умолчанию общий диапазон."""
        lo, hi = self.common_frame_range()
        start = lo if start is None else int(start)
        end = hi if end is None else int(end)
        if end < start:
            raise ValueError(f"пустой диапазон кадров: {start}..{end}")
        chosen = self.cameras if names is None else [c for c in self.cameras if c.name in names]
        return [TruckVideoSource(cam, start=start, end=end, fps=self.fps) for cam in chosen]

    # ---------- manifest ----------

    def manifest(self, *, with_hashes: bool = True) -> dict:
        """Полный manifest набора: файлы, свойства, диапазоны, допущения."""
        start, end = self.common_frame_range()
        cams = []
        for cam in self.cameras:
            frames = self.csv_frames(cam.name)
            entry = {
                "name": cam.name,
                "index": cam.index,
                "video": str(cam.video).replace("\\", "/"),
                "camera_csv": str(cam.csv).replace("\\", "/"),
                **self._props[cam.name],
                "csv_frames": len(frames),
                "csv_first": frames[0] if frames else None,
                "csv_last": frames[-1] if frames else None,
                "static": self.provider.is_static(cam.name),
                "video_bytes": cam.video.stat().st_size,
                "csv_bytes": cam.csv.stat().st_size,
            }
            if with_hashes:
                entry["video_sha256"] = sha256_file(cam.video)
                entry["camera_csv_sha256"] = sha256_file(cam.csv)
            cams.append(entry)
        gt = self.ground_truth()
        gt_entry = {
            "path": str(self.gt_csv).replace("\\", "/"),
            "frames": int(gt.frames.size),
            "first": int(gt.frames.min()) if gt.frames.size else None,
            "last": int(gt.frames.max()) if gt.frames.size else None,
            "invalid_cells": gt.invalid_cells,
            "repaired_frames": gt.repaired_frames,
            "bytes": self.gt_csv.stat().st_size,
        }
        if with_hashes:
            gt_entry["sha256"] = sha256_file(self.gt_csv)
        return {
            "dataset": "pelenga_truck",
            "root": str(self.root).replace("\\", "/"),
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "cameras": cams,
            "ground_truth": gt_entry,
            "common_frame_range": [start, end],
            "common_frames": end - start + 1,
            "bounds_note": (
                "Обрабатывается пересечение кадров видео и строк CSV. "
                "Строка camera CSV с номером >= video_frame_count видеокадра не имеет "
                "и не обрабатывается (off-by-one исключён)."
            ),
            "units": {
                "camera_csv": "сантиметры UE (world)",
                "gt_csv": "сантиметры UE (world)",
                "library": "метры (CameraCalibration.unit == 1.0)",
            },
            "assumptions": [
                "Единицы — Unreal centimeters -> meters (x0.01).",
                "Правая система: X_ue=X_u, Y_ue=-Y_u, Z_ue=Z_u (S_FLIP, det=-1).",
                "Локальные оси: UE X forward / Y right / Z up -> OpenCV x right, y down, z forward.",
                "Эмпирическая поправка: знак pitch CSV зеркален относительно рендера.",
                "Intrinsics: pinhole, fx=focal*width/sensor_w, fy=focal*height/sensor_h, "
                "principal point в центре, дисторсия 0. Не подтверждено организаторами.",
                "Anchor детекции (bbox center) не равен actor pivot — см. docs/OPEN_QUESTIONS.md.",
            ],
            "repairs": (
                None if self.repairs_path is None
                else str(self.repairs_path).replace("\\", "/")
            ),
        }

    def write_manifest(self, path: str | Path, *, with_hashes: bool = True) -> dict:
        doc = self.manifest(with_hashes=with_hashes)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        return doc


class TruckVideoSource:
    """Последовательный читатель одного MP4 набора «Пеленга».

    Таймкод — ``frame / fps``, где ``frame`` — номер кадра, совпадающий с
    колонкой ``Frame`` в CSV камер. Читает кадры ``start..end`` включительно и
    завершается (``read() -> None``) на первом кадре за пределами диапазона
    или при ошибке декодирования. Отдельного PTS из контейнера OpenCV не
    отдаёт, поэтому синхронность по номеру кадра — рабочее допущение
    (вопрос №6 организаторам, docs/OPEN_QUESTIONS.md).
    """

    def __init__(self, camera: TruckCamera, *, start: int = 0, end: int | None = None,
                 fps: float = 30.0) -> None:
        self.camera = camera
        self.name = camera.name
        self.start = int(start)
        self.end = end
        self.fps = float(fps)
        self._cap = cv2.VideoCapture(str(camera.video))
        if not self._cap.isOpened():
            raise RuntimeError(f"не удалось открыть видео: {camera.video}")
        if self.start > 0:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, self.start)
        self._frame = self.start
        self.frames_read = 0
        self._closed = False

    @property
    def frame_index(self) -> int:
        """Номер последнего прочитанного кадра (= Frame в CSV)."""
        return self._frame

    @property
    def closed(self) -> bool:
        return self._closed

    def read(self) -> tuple[float, np.ndarray, None] | None:
        if self._closed:
            return None
        if self.end is not None and self._frame > self.end:
            return None
        ok, frame = self._cap.read()
        if not ok:
            return None
        stamp = self._frame / self.fps
        self._frame += 1
        self.frames_read += 1
        return stamp, frame, None

    def close(self) -> None:
        if not self._closed:
            self._cap.release()
            self._closed = True

    def __repr__(self) -> str:  # pragma: no cover - диагностика
        return f"TruckVideoSource({self.name}, frames {self.start}..{self.end})"
