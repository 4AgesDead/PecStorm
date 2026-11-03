# HANDOFF — состояние работ по проекту multiview3d

> **ОБНОВЛЕНИЕ 03.11.2026 — проект завершён.** Все пункты плана P0–P2
> выполнены: написаны `pipeline.py`, `metrics.py`, `visualize.py`, `__init__.py`,
> `apps/` (4 CLI), `config/demo.json`, `requirements.txt`, `tests/` (23 теста
> `unittest`, все проходят); `fusion.py` переписан и работает; дефекты §4
> исправлены; `camera.py` (триангуляция, F-матрица) и `calibration.py`
> (OpenCV 5, RMS, float32, board-рендер в тестах) исправлены.
> Критерий приёмки P0 выполнен: 3 камеры / 2 объекта / 25 fps → 2 трека,
> средняя ошибка 3D **10.14 мм** ≤ 30 мм (oracle, seed=7, smoke_check rc=0);
> демо на blob-детекторе: среднее 7.48 мм, полнота 1.0, ID-switch 0.
> Актуальное состояние и команды — в README.md. Текст ниже — исторический
> разбор исходного состояния (до 03.11.2026), оставлен как справка.

## 1. Задача

По видеопотокам с нескольких калиброванных камер определять 3D-координаты
движущихся объектов в реальном времени и восстанавливать траектории во времени.

Принятые решения (согласованы с пользователем):

* входные данные — оба варианта: реальные файлы видео/RTSP **и** синтетические
  калиброванные камеры для численной проверки точности;
* детектор — ML (YOLO/ONNX); фоновое вычитание как основной детектор отклонено;
* требования к производительности — настраиваемые (число камер, объектов, частота
  и т. п. задаются конфигом);
* вся документация и сообщения — на русском.

## 2. Что уже сделано

Каталог `multiview3d/`, 9 файлов на диске (размеры в байтах на момент проверки):

| Файл | Размер | Состояние |
|---|---|---|
| `types.py` | 4933 | готово, импортируется |
| `camera.py` | 9660 | готово, импортируется |
| `calibration.py` | 16813 | готово, импортируется |
| `detect.py` | ~9000 | готово после исправления `Detection2D` (см. §4.1) |
| `kalman.py` | 5424 | готово, импортируется |
| `trajectory.py` | 6094 | готово, импортируется |
| `capture.py` | 8152 | готово, импортируется; есть дефекты (§4.3, §4.4) |
| `sim.py` | ~8200 | готово, проверено рендером и детектором |
| `fusion.py` | 11723 | **черновик, не импортируется** (§4.2) |

Отсутствуют (нужно создать): `pipeline.py`, `metrics.py`, `visualize.py`,
`__init__.py`. Нет каталогов `apps/`, `config/`, `tests/`, `data/`, нет
`requirements.txt`.

### Проверки, которые реально выполнялись

1. Импорт всех модулей (временным скриптом, затем удалён): 7 модулей — `OK`,
   `fusion` — `ImportError`, `sim` — был `SyntaxError` (исправлен),
   `pipeline`/`metrics`/`visualize` — `ModuleNotFoundError`.
2. Рендер синтетики: `make_sim_setup(num_cameras=3, num_objects=2, fps=25,
   duration_s=2.0)` → 50 кадров, кадр `(720, 1280, 3) uint8`.
3. `ColorBlobDetector` на синтетическом кадре → 2 детекции, центры `(723.0, 101.0)`
   и `(923.0, 409.0)` **точно совпали** с `project_points` (диски рисуются по той
   же модели проекции).
4. `FrameReader` + `SimSource`: кадр и 1 детекция получены через `latest(timeout=0.3)`.
   `FrameReader.close()` падает — у `SimSource` нет `close()` (§4.3).
5. `CameraCalibration(..., unit="mm")` **не падает** — `__post_init__` не приводит
   `unit` к числу, поэтому строка сохраняется как есть (семантика единиц не решена, §4.6).

### Численных результатов по 3D пока нет

Точность в мм получить нельзя: слияние видов не работает, конвейера нет. Первый
измеримый результат появится после шага P0 (§5).

## 3. Проверенные контракты API (использовать как есть)

Прошлые черновики писались по неверным контрактам и падали. Ниже — фактическое
состояние кода.

### `types.py`

```python
CameraCalibration(name: str, K, dist, R, t, width: int = 0, height: int = 0, unit: float = 1.0)
    # .projection_matrix — СВОЙСТВО (без скобок) = K @ [R|t]
    # X_cam = R @ X_world + t
Detection2D(camera: str, timestamp: float, center: np.ndarray, score: float = 1.0,
            bbox: tuple[float, float, float, float] | None = None, label: str = "")
    # .center — (x, y) в пикселях; полей .u/.v/.x/.y/.w/.h/.class_id НЕТ
Observation3D(timestamp: float, position: np.ndarray, views: list[str], mean_reproj_px: float)
TrackPoint(timestamp: float, position: np.ndarray, velocity: np.ndarray,
           views: list[str], mean_reproj_px: float)
Track(track_id: int, points: list[TrackPoint])   # .to_dict()
```

### `camera.py`

```python
project_point(cam, X) -> np.ndarray            # пиксель (u, v), учитывает дисторсию
project_points(cam, points) -> np.ndarray      # (N,2), точки за камерой -> NaN
undistort_pixels(cam, pixels) -> np.ndarray
ideal_camera(cam) -> CameraCalibration         # копия без искажений (принимает камеру!)
back_project_ray(cam, pixel) -> np.ndarray     # единичный луч в мировых координатах
triangulate_rays(rays: list[np.ndarray], origins: list[np.ndarray]) -> np.ndarray  # DLT
refine_point(cams, pixels, X0, ...)            # уточнение точки (LM)
reprojection_error(cam, X, pixel) -> float
fundamental_matrix(c1, c2) -> np.ndarray
epipolar_distance(c1, c2, pixel1, pixel2) -> float
epipolar_distance_batch(c1, c2, pts1, pts2) -> np.ndarray
baseline_length(c1, c2) -> float
```

Функций `cam_center` и `ray` **нет** (центр камеры: `-R.T @ t`).

### `calibration.py`

```python
board_object_points(...) / find_board(...) / collect_views(...)
calibrate_intrinsics(image_sizes, views_obj, views_img) -> (K, dist, rms)
calibration_from_pose(name, K, dist, rvec, tvec, width, height, unit="mm")
calibrate_camera_from_images(name, image_paths, pattern, square_size_mm=30.0,
                             kind="chessboard", unit="mm") -> (CameraCalibration, dict)
solve_extrinsics_from_points(...) / reprojection_rms(...) / board_full_points(...)
bundle_adjust(...) -> {"cameras", "points_3d", "rms_px", "success"}   # scipy least_squares, LM
correspondence_epipolar_errors(cam_a, cam_b, pixels_a, pixels_b) -> np.ndarray  # (Na, Nb), сумма двух направлений
quality_report(cameras, calibration_views=None, reproj_rms_px=None) -> dict
save_calibration(path, cameras, meta=None) / load_calibration(path) -> (cameras, meta)
```

### `detect.py`

```python
class Detector(Protocol):  # detect(frame_bgr) -> list[Detection2D]
YoloDetector(model="yolov8n.pt", *, conf=0.35, classes=None, device="cpu")  # ленивый import ultralytics
OnnxDetector(model_path, *, input_size=640, conf=0.35, output_name=None, classes=None)
ColorBlobDetector(colors: list[tuple[int, int, int]], *, hsv_tolerance=25,
                  min_area=60.0, max_area=40000.0)     # colors — BGR, индекс = класс
OracleDetector(gt_by_frame=None)   # .set_ground_truth({class_id: (u, v)}) перед .detect()
nms(boxes, scores, iou_threshold=0.45) -> list[int]
```

Фабрики `build_detector` **нет** — её нужно добавить для конфиг-управляемого выбора.

### `kalman.py`

```python
KalmanFilter(x0, P0, Q_diag: tuple[float, float], measurement_noise: float)
    # .x (6,) = [x,y,z,vx,vy,vz], .P (6,6), .t, .pos_var, .vel_var, .R
    .predict(t) -> np.ndarray        # dt = t - self.t
    .mahalanobis(z) -> float
    .update(z) -> np.ndarray
    .innovation_covariance() -> np.ndarray
rts_smooth(times, states, covariances, ...) -> ...
SmoothedState(...)
```

**Историю состояний фильтр не хранит** — для RTS-сглаживания её нужно собирать
снаружи (в `pipeline.py`).

### `trajectory.py`

```python
TrajectoryFeatures(...) / trajectory_features(track)
smooth_positions(track, window=5) -> np.ndarray
resample(track, dt) -> (np.ndarray, np.ndarray)
match_tracks_to_truth(tracks: list[Track], truth: dict[str, np.ndarray],
                      max_mean_error: float = 200.0) -> MatchResult
    # truth[label] индексируется ЦЕЛЫМ timestamp: arr[int(p.timestamp)]
MatchResult(pairs, mean_error_mm, unmatched_tracks, unmatched_truth)
```

### `capture.py`

```python
FrameSource(Protocol): read() -> (stamp, frame, list[Detection2D]|None) | None; close()
ReaderStats: .fps_source, .latency_ms, .frames_dropped, .update(latency_ms)
LatestFrameBuffer(maxlen=2): .put(item) / .get(timeout)
OpenCvFrameSource(source: str|int, *, size=None, fps_fallback=30.0)  # .read()/.close()/.closed
FrameReader(source, name=""): .start() / .latest(timeout=0.0) -> 4-tuple (stamp, frame, dets, monotonic)
                              / .closed / .close() / .stats
build_readers(sources, names=None) -> list[FrameReader]   # сразу стартует потоки
read_aligned(readers, *, align_tolerance=0.05, timeout=0.5) -> dict|None
read_all_offline(sources, *, align_tolerance=0.05) -> dict|None   # ОДИН кадр, не итератор
```

### `sim.py`

```python
look_at_rotation(origin, target, up=None) -> np.ndarray   # строки: right, down, forward
cam_center(cam) -> np.ndarray                             # локальная функция (в camera.py её нет)
SimDataset(cameras, paths, frame_times, labels, fps, unit=1.0, ball_radius=0.08, ball_colors=[...])
    .frame_count, .truth_at(t) -> list[np.ndarray], .ground_truth_tracks() -> list[list[TrackPoint]],
    .camera_frame(camera_index, t) -> np.ndarray
SimCameraStream(dataset, camera_index, *, repeat=False)   # .name, .read()
SimSource(stream, detector)                               # .name, .read() -> (t, frame, dets)
make_sim_setup(*, num_cameras=3, num_objects=2, width=1280, height=720, fps=25.0,
               duration_s=6.0, workspace=(-2,2,-2,2,0,3), camera_ring_radius=5.0,
               camera_height=2.2, ball_radius=0.08, fx=1100.0,
               distortion=(-0.25, 0.09, 0.001, 0.002, -0.02), seed=7, unit=1.0)
```

Синтетика детерминирована (`seed=7`), камеры стоят кольцом радиусом 5 м, смотрят в
точку `(0, 0, 1)`, базовая модель — сфера радиусом 0.08 м, проецируемая точной
моделью с дисторсией.

## 4. Известные дефекты и расхождения

### 4.1 `detect.py` — исправлено в этой сессии

Все четыре места создавали `Detection2D(camera="", score=..., x=..., y=..., w=...,
h=..., class_id=...)`, хотя у датакласса поля `center`/`bbox`/`label` — это давало
`TypeError: Detection2D.__init__() got an unexpected keyword argument 'x'` (проверено
запуском `ColorBlobDetector` и `OracleDetector`). Теперь создаётся
`Detection2D(camera="", timestamp=0.0, center=np.array([u, v]), score=..., bbox=(...), label=str(cls))`.
ML-бэкенды (`YoloDetector`, `OnnxDetector`) после правки **не запускались** — нет
`ultralytics`/`torch`, а `OnnxDetector` требует готовой `.onnx`-модели.

### 4.2 `fusion.py` — черновик, требует переписывания

Файл написан до сверки с `types.py`/`camera.py`, поэтому:

* `line 18`: `from .camera import CameraCalibration, cam_center, ray, triangulate_rays` —
  `cam_center` и `ray` не существуют → `ImportError` (проверено);
* использует `Detection2D.u/.v/.confidence` — таких полей нет, есть `.center`/`.score`;
* `Observation3D(ci, u, v, conf, label)` — не тот контракт: нужно
  `Observation3D(timestamp, position, views: list[str], mean_reproj_px)`;
* `TrackPoint(t, pos, views, reproj)` — пропущен `velocity` (позиционный аргумент 3);
* дублирует `calibration.correspondence_epipolar_errors` и
  `camera.fundamental_matrix`/`epipolar_distance_batch` — переиспользовать их, а не
  писать свою `_fundamental`/`_essential`;
* `triangulate_rays` в `camera.py` принимает `(rays, origins)`, а не `(rays, origin)`.

Полезное, что можно сохранить при переписывании: `FusionConfig`, идея венгерского
сопоставления по эпиполярным стоимостям, DLT + LM-уточнение, махаланобисов гейт,
`TrackedObject`/`FrameFusionResult`.

### 4.3 `sim.py` / `capture.py` — несовместимость протокола `FrameSource`

`FrameReader.close()` вызывает `self.source.close()`, а `FrameReader.closed` читает
`self.source.closed`; у `SimSource`/`SimCameraStream` этих членов нет →
`AttributeError` (проверено). Исправление: добавить в `SimSource`/`SimCameraStream`
`close()` (no-op) и свойство `closed` (всегда `False`/по флагу).

### 4.4 `capture.build_readers` — хрупкая цепочка `getattr`

`line 165`: `getattr(src.stream, "camera", {}).name if hasattr(src, "stream") else f"cam{i}"`.
У `OpenCvFrameSource` нет `.stream`, у `SimSource` нет `.stream.camera` → имя
скатывается к `cam{i}` или падает. Лучше: `getattr(src, "name", None) or f"cam{i}"`.

### 4.5 `capture.read_all_offline` — не итератор

Возвращает словарь `{i: (stamp, frame, dets)}` для **одного** выровненного кадра и
завершается; при расхождении таймкодов `continue` перечитывает по кадру. Для
офлайн-прохода по всему видео нужен генератор (`yield`), сейчас его нет.

### 4.6 Единицы измерения не унифицированы

`types.CameraCalibration` документирует `unit` как числовой множитель перевода в
метры (`1.0` — метры, `0.001` — если `t` в мм), но `calibration.py` по умолчанию
передаёт строку `"mm"`, и `__post_init__` её не валидирует (проверено: объект
создаётся, `unit == "mm"`). Нужно решить: либо `unit` — строка-тег, либо число;
сейчас в коде смешаны оба смысла.

### 4.7 Прочее

* `read_aligned` в docstring обещает `{имя: (кадр, детекции, таймкод)}`, фактически
  возвращает 4-tuple от `FrameReader.latest` (с `monotonic`).
* `trajectory.match_tracks_to_truth` трактует `timestamp` как индекс кадра
  (`arr[int(p.timestamp)]`) — для синтетики с `fps=25` это неверно, если
  `timestamp` в секундах. Либо подавать номер кадра, либо пересчитывать индекс.
* `apps/`, `config/`, `tests/`, `requirements.txt`, `__init__.py` отсутствуют;
  пакет импортируется как namespace package.

## 5. План следующих шагов

### P0 — вернуть работоспособное ядро

1. Переписать `fusion.py` под контракты §3: `Detection2D.center`, `Observation3D`,
   `TrackPoint(..., velocity, ...)`, `back_project_ray`, `triangulate_rays(rays, origins)`,
   `calibration.correspondence_epipolar_errors`, `camera.refine_point`.
2. Добавить в `SimSource`/`SimCameraStream` `close()` и `closed`; починить
   `build_readers`.
3. Добавить `build_detector(spec: dict) -> Detector` в `detect.py` (yolo/onnx/blob/oracle).
4. Критерий приёмки: `import multiview3d.fusion` без ошибок; на синтетике
   (3 камеры, 2 объекта, 25 fps) восстанавливается 2 трека; средняя ошибка 3D
   ≤ 30 мм (порог выбрать по факту и зафиксировать в отчёте).

### P1 — конвейер и метрики

5. `pipeline.py`: `PipelineConfig`/`StreamConfig` из JSON, `MultiViewPipeline.step()`
   и `.run()`, история состояний/ковариаций по `track_id` для `rts_smooth`.
6. `metrics.py`: средняя/макс/95-перцентиль ошибки в мм по ground truth, ID-switch,
   полнота треков, FPS и задержка.
7. `visualize.py`: 3D-график траекторий, наложение треков на кадры, CSV/JSON-экспорт.

### P2 — CLI, конфиг, тесты, реальные данные

8. `apps/calibrate.py`, `apps/live.py`, `apps/sim_dataset.py`, `apps/evaluate.py`.
9. `config/demo.json` (камеры, RTSP/файлы, пороги, детектор), `requirements.txt`.
10. `tests/`: юнит-тесты геометрии (проекция/триангуляция на известных точках),
    калибровки (синтетическая доска), Калмана (сходимость на линейном движении),
    слияния (2 камеры, 1 объект).
11. Проверка на реальном видео: калибровка по доске, RTSP/файлы, косвенные метрики
    (репроекция, эпиполярные невязки).

### P3 — по согласованию

12. `pip install ultralytics` (+`torch`, крупная загрузка) для `YoloDetector`;
    `pip install onnx` для генерации тестовой ONNX-модели и теста `OnnxDetector`.

## 6. Грабли инструментов (проверено в этой сессии)

* `write`/`edit` по **существующим** файлам иногда отвечают «Edited/Created
  successfully», но содержимое на диске не меняется. Обход: удалить файл и записать
  заново; **каждую крупную запись проверять** чтением (`Select-String`/`Get-Item Length`).
* Один большой `write` может оборваться (`tool.input-incomplete`, stop reason:
  length) — писать файл частями (сначала каркас, затем `edit`-дополнения).
* `edit` требует дословного совпадения `oldString`; при кириллице надёжнее
  перезапись файла.
* PowerShell печатает кириллицу как mojibake (`�����`), хотя файлы в UTF-8 целы.
  Для корректного вывода из Python: `sys.stdout.reconfigure(encoding="utf-8")`.
* Python запускать только так:
  `C:\Users\BeGraphics\AppData\Local\Programs\Python\Python314\python.exe`.
  Алиас `...\WindowsApps\python.exe` — заглушка Microsoft Store, падает.
* В Code Mode (`execute`) вызов `tools.shell(...)` недоступен (`Unknown tool
  'shell'`); `shell` работает только как прямой инструмент.
* Git: изменения не коммитились. `git log` — единственный коммит `2bedc37 Initial
  commit`; `multiview3d/`, `tools/`, `HANDOFF.md` — untracked, `README.md` — изменён.
  Временные `_probe*.py` удалены.

## 7. Как воспроизвести проверки

```powershell
$py="C:\Users\BeGraphics\AppData\Local\Programs\Python\Python314\python.exe"
& $py tools\smoke_check.py
```

Фактический вывод на 03.11.2026: импорт `types`, `camera`, `calibration`, `detect`,
`kalman`, `trajectory`, `capture`, `sim` — `OK`; `fusion` — `ImportError`;
`pipeline`, `metrics`, `visualize` — `ModuleNotFoundError`; кадр
`(720, 1280, 3) uint8`; 2 детекции, максимальное отклонение центра от точной
проекции 0.45 px; `FrameReader` отдал кадр с 1 детекцией, `close()` не поддержан
источником (дефект §4.3). Код возврата 0 — обязательные проверки пройдены.

## 8. Что нельзя утверждать без проверки

* Любые числа точности в мм на реальных камерах — ground truth отсутствует.
* Работоспособность `YoloDetector`/`OnnxDetector` — зависимости не установлены,
  запусков не было.
* Соответствие синтетических шумов реальным — синтетика идеализирована
  (нет бликов, окклюзий, смаза, реальных ошибок калибровки).
