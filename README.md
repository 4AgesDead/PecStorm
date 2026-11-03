# multiview3d — 3D-координаты и траектории объектов по нескольким камерам

Библиотека на Python: по видеопотокам с нескольких **калиброванных** камер находит
движущиеся объекты, восстанавливает их 3D-координаты (триангуляция по нескольким
видам) и ведёт траектории во времени фильтром Калмана.

Задача сформулирована так: «по видеопотокам с нескольких калиброванных камер
определять 3D-координаты движущегося объекта (объектов) в режиме реального времени
и восстанавливать его траекторию во времени».

> **Статус: проект завершён, сквозной прогон работает.** Реализованы и проверены
> запуском все модули ядра (12), конвейер `pipeline.py`, метрики, визуализация,
> CLI-приложения (`apps/`), конфиг и тесты (70 тестов `unittest`, все зелёные).
> Критерий приёмки выполнен: синтетика 3 камеры / 2 объекта / 25 fps → ровно
> 2 трека, средняя ошибка 3D после RTS-сглаживания ≤ 30 мм (факт. 10.1 мм).
> Исторический разбор исходного состояния — в [HANDOFF.md](HANDOFF.md).
> Быстрый гайд «что запускать» — [docs/QUICKSTART.md](docs/QUICKSTART.md).

## Что проверено запуском (2026-11-03)

| Проверка | Результат |
|---|---|
| Импорт всех 12 модулей `multiview3d` | успешно (`tools/smoke_check.py`, rc=0) |
| Рендер синтетической сцены (3 камеры, 1280×720) | кадр `(720, 1280, 3) uint8` |
| Детектор цветных меток на синтетическом кадре | 2 метки, макс. отклонение центра 0.45 px |
| `FrameReader` + `close()/closed` на синтетическом источнике | кадр и детекции получены, источник закрыт |
| Калибровка по синтетической шахматной доске (8 ракурсов, OpenCV 5) | RMS 0.07 px; fx/fy восстановлены в пределах 0.3%, cx/cy — в 2 px |
| Сквозной офлайн-прогон (oracle-детектор, 3 камеры, 2 объекта, 2 с, seed=7) | 2 трека, ID-switch 0, среднее 10.14 мм, p95 27.3 мм, max 42.8 мм (90 точек) |
| Демо-прогон `apps.sim_dataset` (blob-детектор, 3 камеры, 2 объекта, 6 с, seed=7) | 2 трека, полнота 1.0/1.0, ID-switch 0, среднее 7.48 мм (300 точек), ~260 fps обработки |
| `python -m unittest discover -s tests` | 70 тестов, OK |

Эталонный критерий приёмки (HANDOFF, P0): **3 камеры, 2 объекта, 25 fps →
2 трека, средняя ошибка ≤ 30 мм** — выполнен с запасом.

## Установка

Проверено на Windows, Python **3.14.10**, интерпретатор
Фактически установленные и проверенные версии:

| Пакет | Версия |
|---|---|
| numpy | 2.5.4 |
| opencv-python-headless (`cv2`) | 5.0.0 |
| scipy | 1.18.1 |
| matplotlib | 3.11.2 |
| onnxruntime | 1.31.1 |

ML-бэкенды: `YoloDetector` требует `ultralytics` (+ `torch`) — **не установлены**;
`OnnxDetector` требует только `onnxruntime` (установлен). Пакет `onnx` (для
генерации тестовых моделей) отсутствует. Рабочие детекторы без ML —
`ColorBlobDetector` и `OracleDetector` (по ground truth, только для валидации).

## Быстрый старт (то, что работает сейчас)

Проверка ядра одной командой:

```powershell
& path python tools\smoke_check.py
```

Скрипт печатает статус импорта всех модулей, проверяет рендер синтетики,
совпадение детекций с точной проекцией, работу `FrameReader` и сквозной
офлайн-прогон (критерий приёмки: 2 трека, средняя ошибка ≤ 30 мм);
код возврата 1 — если обязательная проверка провалена.

Синтетическая сцена с точно известной калибровкой + детектор меток:

```python
import numpy as np
from multiview3d.sim import make_sim_setup, SimCameraStream, SimSource
from multiview3d.detect import ColorBlobDetector

dataset = make_sim_setup(num_cameras=3, num_objects=2, fps=25, duration_s=2.0)
stream = SimCameraStream(dataset, camera_index=0)
t, frame, _ = stream.read()                 # (таймкод, кадр BGR, детекции|None)

detector = ColorBlobDetector(dataset.ball_colors[:len(dataset.paths)])
detections = detector.detect(frame)         # list[Detection2D]
print(t, frame.shape, [(d.label, np.round(d.center, 1)) for d in detections])
print(dataset.truth_at(t))                  # ground truth 3D, метры

source = SimSource(stream, detector)        # источник для FrameReader
print(source.name)
```

Ручная проверка геометрии одной камеры:

```python
from multiview3d.camera import project_points, back_project_ray
from multiview3d.types import CameraCalibration

cam = dataset.cameras[0]
pixels = project_points(cam, dataset.truth_at(t))   # мировые точки -> пиксели
ray = back_project_ray(cam, pixels[0])              # пиксель -> единичный луч
```

## Архитектура

```
видео / RTSP ─┐
              ├─ capture.FrameReader (поток на камеру, буфер «последний кадр»)
синтетика ────┘        │
                       ▼
             detect.Detector.detect(frame)  -> Detection2D (пиксели)
                       │
                       ▼
   fusion: эпиполярные стоимости -> венгерское сопоставление видов
           -> DLT-триангуляция -> уточнение по репроекции (LM)
                       │
                       ▼
   fusion.MultiViewTracker: ассоциация по махаланобису + KalmanFilter (CV-модель)
                       │
                       ▼
   trajectory / metrics / visualize: траектории, ошибки в мм, CSV/JSON, 3D-графики
```

### Модули (`multiview3d/`)

| Файл | Назначение | Состояние |
|---|---|---|
| `types.py` | `CameraCalibration`, `Detection2D`, `Observation3D`, `TrackPoint`, `Track` | готово |
| `camera.py` | проекция с искажениями, обратные лучи, DLT-триангуляция, уточнение точки, эпиполярные расстояния, фундаментальная матрица | готово |
| `calibration.py` | калибровка по шахматной доске/кругам (`cv2.calibrateCamera`), внешние параметры через `solvePnP` в системе полигона, `bundle_adjust` (scipy LM), отчёт качества, сохранение/загрузка JSON | готово |
| `detect.py` | протокол `Detector`, `YoloDetector`, `OnnxDetector` (+ свой NMS), `ColorBlobDetector`, `OracleDetector`, `build_detector` | готово (ML-бэкенды не запускались — `ultralytics` не установлен) |
| `kalman.py` | `KalmanFilter` (6D: позиция + скорость, махаланобисов гейт), `rts_smooth` | готово |
| `trajectory.py` | признаки траектории, сглаживание, ресемплинг, сопоставление треков с эталоном | готово |
| `capture.py` | `FrameSource`, `FrameReader` (поток), `LatestFrameBuffer`, `OpenCvFrameSource`, `read_aligned`, `read_all_offline`, `read_first_aligned` | готово (дефекты из HANDOFF исправлены) |
| `sim.py` | синтетический полигон: камеры с известными K/R/t/дисторсией, рендер, ground truth, `SimSource`, `OracleSimSource` | готово |
| `fusion.py` | сопоставление видов (венгерский алгоритм по эпиполярным стоимостям), триангуляция + уточнение LM, 3D-трекинг | готово, проверено сквозными тестами |
| `pipeline.py` | оркестрация, JSON-конфиг (`PipelineConfig`), офлайн/реалтайм режимы, история состояний для RTS | готово |
| `metrics.py` | ошибки в мм, полнота/точность треков, ID-switch, частота обработки; `evaluate_truck` — MAE/RMSE/P95/P99 + покрытие по статусам | готово |
| `visualize.py` | 3D-графики траекторий, наложение треков на кадр, CSV/JSON экспорт-импорт; `plot_components_vs_time` — график X/Y/Z «эталон vs оценка» | готово |
| `sources.py` | `DetectingFrameSource` — детектор поверх любого `FrameSource` (дефект P0.3) | готово |
| `detect.py` | `YoloDetector` (ultralytics YOLOv8, классы COCO `truck/car/bus/train`) | готово, полный прогон 5005 кадро-камер |
| `unreal_camera.py` | импорт CSV Unreal (CP1251), конверсия единиц/осей, GT с `invalid_cells` и sidecar-репарами | готово |
| `camera_provider.py` | `CameraStateProvider`: поза камеры по номеру кадра, статичные cam1–4, вращение cam5 | готово |
| `truck_dataset.py` | датасет «Пеленга»: discover/manifest/общая область кадров/`TruckVideoSource` | готово |
| `truck_pipeline.py` | стадии truck-workflow: детекция → CSV → реконструкция → CSV/JSON оценок | готово |

## Команды (реализованы, проверены)

```
py tools/smoke_check.py                                # быстрая проверка ядра
py -m unittest discover -s tests                       # тесты (pytest не требуется)
py -m apps.sim_dataset --config config/demo.json --out out          # синтетика -> треки/метрики/графики
py -m apps.evaluate  --tracks out/tracks.csv --truth out/truth.csv  # оценка по CSV
py -m apps.calibrate --images data/board --pattern 9x6 --square 30 --out config/calib.json
py -m apps.live      --config config/demo.json --record out/tracks.csv
```

`apps.sim_dataset` пишет `tracks.csv`, `tracks.json`, `truth.csv`, `metrics.json`
и `trajectories.png`; `apps.evaluate` читает эти CSV и печатает отчёт.
Проверено: `sim_dataset` + `evaluate` на `config/demo.json` (rc=0),
`calibrate`/`live` — запуск с `--help` (импорт и аргументы валидны; для
полного прогона нужны реальные изображения полигона / камеры).

## Реальные данные «Пеленга» (truck-workflow)

Помимо синтетики, пакет обрабатывает выданный набор: 5 MP4 (1920×1080, 30 fps,
кадры 0..1000), 5 camera CSV (Unreal, CP1251) и GT-траектория цели.
Подробности формата, конверсии и ограничений — [docs/TRUCK_DATASET.md](docs/TRUCK_DATASET.md);
вопросы организаторам — [docs/OPEN_QUESTIONS.md](docs/OPEN_QUESTIONS.md).

```
py apps/import_truck_dataset.py                        # manifest + отчёт о расхождениях
py apps/detect_truck.py --start 0 --end 1000 --step 1 `
    --model yolov8x.pt --conf 0.25 --imgsz 960             # ML-детектор (YOLO, класс truck)
py tools/anchor_fit_views.py --detections out/truck/detections.csv `
    --max-reprojection-px 50                               # калибровка якоря по наборам видов
py apps/reconstruct_truck.py --detections out/truck/detections.csv
py apps/evaluate_truck.py --estimates out/truck/estimates.csv
py apps/visualize_truck.py --estimates out/truck/estimates.csv `
    --frames 83,190,400,600,834,924 --out out/truck/viz --video
py tools/plot_components.py --estimates out/truck/estimates.csv  # график X/Y/Z: эталон vs оценка
py tools/make_report.py                                # сводка + все графики
```

Графики: `out/truck/report/components_vs_time.png` («Итоговый результат»:
`X_t`/`Y_t`/`Z_t` сплошными, `X_det`/`Y_det`/`Z_det` штриховыми),
`components_error_vs_time.png` (то же + расхождение «оценка − эталон» по осям),
`error_vs_frame.png`, `trajectory_xy.png`, `coverage.png`.

Итоговые числа полного прогона — [docs/RESULTS.md](docs/RESULTS.md)
(детекция 93.7 % кадров, покрытие 61.3 %, MAE 4.28 м при медиане 2.9 м;
in-sample по калибровке якоря, parity-CV — 4.5 м).

Что важно знать перед чтением результатов:

* позы камер обновляются **на каждом кадре** (cam5 вращается), статичные
  cam1–4 кэшируются;
* детектор — ML (YOLOv8, класс COCO `truck`), фоновое вычитание как основной
  детектор не используется;
* 2D-якорь (центр bbox) не равен GT-pivot: расхождение 57–229 px по камерам.
  Компенсируется 3D-сдвигом **на набор видов** (`anchor_offset_views`,
  `tools/anchor_fit_views.py`); он проверен разбиением чётные/нечётные кадры и
  по времени. Общий сдвиг `anchor_offset_m` и 2D-сдвиг по камерам
  (`anchor_offset_px`) реализованы, но на holdout не переносятся — см. §5
  `docs/RESULTS.md`;
* оценки 3D содержат `status`/`method`/`camera_ids` (DTO §11), а ошибка и
  покрытие считаются раздельно (§7.6): «точность только на лёгких кадрах» не
  выдаётся за общий результат;
* GT-кадры 402 и 691 исключены (испорченные Excel-ячейки), исходные файлы не
  изменяются.

Эксперименты: `experiments/visibility_analysis.py` (видимость по камерам),
`projection_sanity.py` (проекция GT-пивота vs детекции),
`detector_benchmark.py` (модель/imgsz → detection rate и время),
`geometry_sweep.py` (пороги слияния → MAE vs coverage),
`robustness_benchmark.py` (шум 2D, ошибки калибровки, сдвиг времени).

## Как измеряется точность

* **Синтетика** — единственный режим, где возможна ошибка в миллиметрах: 3D-пути
  объектов заданы аналитически, камеры имеют точные K/R/t и дисторсию, кадры
  рендерятся той же моделью проекции. Сравнение восстановленных траекторий с
  ground truth даёт среднюю/максимальную ошибку в метрах.
* **Реальные видео** без разметки — ground truth нет, поэтому доступны только
  косвенные метрики: ошибка репроекции в пикселях, эпиполярные невязки,
  стабильность трека, частота и задержка обработки.
* **Датасет «Пеленга»** — GT есть (pivot актора, `xyzpowcameta`), поэтому MAE
  считается в метрах; но он ограничен расхождением «центр bbox ↔ pivot»
  (140 px медиана), см. `docs/RESULTS.md`.

Синтетика проверяет геометрию, сопоставление видов и фильтр, но **не** заменяет
проверку на реальном оборудовании: она ничего не говорит о качестве реального
детектора и о реальных шумах калибровки.

## Ограничения и допущения

* Мировая система координат — система калибровочного полигона; все камеры решаются
  в ней одной задачей, поэтому внешние параметры согласованы между собой.
* `CameraCalibration.unit` — числовой множитель перевода мировых единиц в метры;
  валидируется как float в `__post_init__` (строка вызывает `ValueError`).
  Калибрация по полигону переводит мм в метры сама (`square_size_mm * 0.001`),
  поэтому `unit = 1.0` и мир — метры. Дефект 8 из HANDOFF исправлен.
* Триангуляция устойчива только при достаточном базисе и некомпланарном
  расположении камер; объекты должны быть видны минимум с двух камер.
  Проверено прогонами: при 3 камерах треки стабильны (2 трека, ID-switch 0 при
  seeds 7/42/3-камеры); при 2 камерах потеря видимости ≥2 видов дробит трек на
  фрагменты (замер seed=11: 10 треков) — это ограничение архитектуры
  «минимум 2 вида», а не дефект кода.
* `workspace_bounds` отсекает кандидатов вне объёма; треки объектов, выходящих
  за границы, рвутся (в `config/demo.json` z-границы расширены до ±3 м).
* В реалтайм-режиме (`apps.live`) `FrameReader` пропускает кадры при медленной
  обработке и есть guard `stall_seconds` на остановку потоков; точность
  реалтайм-прогона по синтетике ниже офлайн (замер: 21.8 мм против 7.5 мм) —
  эффект пропуска кадров, а не геометрии.
* Детекция — только ML/специализированный детектор (`Detector`), фоновое
  вычитание как основной метод не используется.
* OpenCV 5.0: `calibrateCamera` возвращает 5 значений (без RMS) и требует
  float32-точки; RMS считается в `calibration.py` самостоятельно по rvecs/tvecs.

## Структура репозитория

```
multiview3d/          библиотека (ядро + truck-workflow: detect, truck_dataset,
                      truck_pipeline, camera_provider, unreal_camera, sources)
apps/                 CLI: import/detect/reconstruct/evaluate/visualize_truck,
                      плюс sim_dataset, evaluate, calibrate, live (синтетика)
config/truck.json     конфиг truck-workflow (5 камер, детектор, пороги слияния)
datasets/truck/       manifest, калибровка якоря, sidecar-репары GT
docs/                 QUICKSTART, TRUCK_DATASET, RESULTS, DETECTION, OPEN_QUESTIONS
experiments/          видимость, проекция, свип порогов, робастность, бенчмарк
tools/                калибровки якоря, отчёт и графики, вспомогательные скрипты
tests/                70 тестов unittest (геометрия, калибровка, fusion,
                      pipeline, truck-workflow, экспорт, графики)
requirements.txt      зафиксированные версии проверенных пакетов
README.md             этот файл
HANDOFF.md            состояние работ, история, грабли инструментов
```

## Лицензия и авторство

Не заданы. Репозиторий создан для хакатона; файл лицензии отсутствует.
