# PECSTORM × SPACE HACKATHON 2.0 — MASTER TECHNICAL HANDOFF

**Назначение документа:** передать другой AI-модели / coding agent полный контекст задания, предоставленных материалов, текущего репозитория и плана работ. Документ рассчитан на модель, которая должна не только дать рекомендации, но и **изучить исходный код и внести изменения в репозиторий PecStorm**.

**Язык проекта и команды:** русский. Имена файлов, классов, функций и API сохранять в английском стиле, как в существующем репозитории.

**Дата исходных материалов:** 3 ноября 2026 года. Сроки ниже взяты из презентации организаторов.

---

# 0. ИНСТРУКЦИЯ ДЛЯ AI-КОДИРОВЩИКА — ПРОЧИТАТЬ ПЕРЕД РАБОТОЙ

Ты выступаешь как senior computer-vision / robotics engineer и coding agent. Твоя задача — адаптировать существующий репозиторий PecStorm к реальному заданию Space Hackathon 2.0 от ОАО «Пеленг», используя приложенные исходники проекта, набор видео/CSV и две презентации.

## 0.1. Что от тебя требуется

1. Изучи **весь** репозиторий, а не только README: `multiview3d/`, `apps/`, `tests/`, `config/`, `requirements.txt`, `HANDOFF.md`.
2. Изучи предоставленный датасет: пять видео, пять CSV камер, CSV с траекторией объекта. Не считай CSV с координатами детекциями: это не так.
3. Изучи обе презентации. Используй формулировки и требования из материалов, не подменяй их общими идеями.
4. Сначала воспроизведи существующие проверки. Зафиксируй исходное состояние и baseline.
5. Затем реализуй приоритетные изменения небольшими, проверяемыми коммитами/этапами. Не переписывай рабочую геометрию без экспериментально подтверждённой причины.
6. Напиши тесты к каждой существенной новой функции.
7. Запусти полный набор тестов, smoke check и интеграционные эксперименты. В итоговом отчёте приведи точные команды и фактические результаты.
8. Обнови документацию: README, инструкции запуска на truck dataset, формат результатов, известные ограничения и критерии качества.

## 0.2. Принципы, которые нельзя нарушать

- **Не используй Ground Truth как вход алгоритма позиционирования.** Файл `object_transform_SM_AA_track.csv` разрешён для разработки, диагностики и offline-оценки только на предоставленном тестовом наборе. Нельзя использовать его координаты для формирования предсказанных координат, выбора гиперпараметров по конкретному кадру теста, подмены детекций или заполнения результата. Если нужен oracle-режим, он должен быть явно помечен как тестовый и не входить в production pipeline.
- Не называй предположения подтверждёнными фактами. В коде и отчёте помечай исходные параметры, inferred values и неизвестные параметры раздельно.
- Не подменяй отсутствие измерения выдуманной координатой. Если объект виден только в одной камере и нет проверенного monocular/model-based метода, возвращай статус `UNOBSERVABLE` / `PREDICTED_ONLY`, неопределённость и метод оценки. Не маскируй прогноз как измеренную 3D-позицию.
- Не считай центр bounding box одинаковой физической точкой грузовика для разных ракурсов. До триангуляции определи согласованную 2D-опорную точку или несколько 2D keypoints, соответствующих одной и той же 3D-точке/модели.
- Не зашивай значения, связанные с конкретным текущим датасетом, в общую геометрию. Создай отдельный адаптер входных данных/конфигурацию.
- Не делай вывод, что больше камер всегда точнее. Проверяй геометрию, видимость, калибровку и качество каждого наблюдения.
- Не утверждай real-time FPS по времени только `tracker.step()`. Нужны end-to-end latency и FPS всей цепочки.
- Не выдавай offline RTS smoothing за результат онлайн-оценки без задержки.
- Сохраняй исходные предоставленные файлы неизменными. Исправления повреждённых CSV-значений делай отдельными sidecar overrides с логом происхождения.
- Не обучай нейросеть с нуля до проверки предобученной модели и реальных ошибок детекции. Сначала baseline, потом fine-tuning при доказанной необходимости.
- Не внедряй новый сложный tracker / новую нейросеть только потому, что они упомянуты в презентации. Проверяй их на датасете и по вычислительному бюджету.

## 0.3. Обязательный порядок действий

**Шаг A — baseline:** запустить `python -m unittest discover -s tests -v`, `tools/smoke_check.py`, текущий `apps.sim_dataset`; записать результаты.

**Шаг B — audit данных:** извлечь `.7z`, проверить число и параметры видео, прочитать CSV с корректной кодировкой, построить временные/визуальные отчёты по камерам и объекту.

**Шаг C — ingest:** добавить устойчивое чтение данных Пеленга, преобразование системы координат и единиц, поддержку статичных камер 1–4 и покадровой позы камеры 5.

**Шаг D — 2D:** подключить детектор к обычным видеоисточникам, получить и сохранить 2D detections/tracks per camera, оценить recall/precision/центр/опорные точки.

**Шаг E — 3D baseline:** выполнить триангуляцию только на кадрах, где одна и та же физическая точка обнаружена минимум в двух камерах; сравнить с GT.

**Шаг F — одиночная камера и потери:** исследовать, нужен ли 3D model/PnP или иной monocular estimator; если нет достоверной модели, корректно маркировать недоступность 3D-наблюдения и использовать ограниченный прогноз трекера.

**Шаг G — benchmark и отчёт:** MAE/RMSE/P50/P95/P99, coverage, reprojection error, track loss/reacquisition, frame drops, latency; сохранить результаты и графики.

**Шаг H — упаковка сдачи:** формировать координатный CSV по точному формату организаторов (формат пока не подтверждён; сделать настраиваемый exporter).

## 0.4. Режим работы с неизвестными требованиями

Если блокирующий факт неизвестен, не придумывай ответ. Реализуй архитектуру с явной настройкой и продолжай то, что можно проверить. В `docs/OPEN_QUESTIONS.md` веди список вопросов Пеленгу. Не останавливай работу только из-за того, что 3D-модель или output schema ещё не выданы.

---

# 1. СУТЬ ЗАДАНИЯ

## 1.1. Формулировка ТЗ из презентации организаторов

В презентации «Зуенок А..pptx», слайд 2, записано:

> Необходимо разработать систему автоматического определения и сопровождения заданного объекта на видео с определением его местоположения (x, y, z – 3D координаты).

На слайде 7 дана базовая последовательность:

**Детектирование → Трекинг → Триангуляция → 3D-координаты.**

На слайде 9 в составе проекта перечислены:
- набор видеоряда с камер;
- документ с координатами;
- 3D-модель объекта.

В слайде 8 показан пример сравнения `X/Y/Z true` и `X/Y/Z det` на временной сетке с шагом 0,5 с. Это пример оформления/результата из презентации, но **не доказательство**, что итоговый формат сдачи обязательно имеет именно такую частоту или названия полей. Формальный output schema нужно уточнить у организаторов.

## 1.2. Корректная интерпретация

Система должна автоматически сопровождать заданный объект на видеопотоках и восстанавливать/оценивать его мировые координаты `x, y, z` во времени.

Основные компоненты:

1. **2D detection:** найти грузовик на каждом видеокадре.
2. **2D tracking / reacquisition:** сопровождать целевой объект по кадрам отдельной камеры, оценивать уверенность, выявлять `LOST`, повторно захватывать объект.
3. **Multi-camera association:** сопоставить наблюдения одной и той же физической точки/объекта между разными камерами и одним моментом времени.
4. **3D reconstruction:** триангуляция при наличии минимум двух корректных ракурсов; при одном ракурсе может понадобиться model-based monocular pose estimation, если предоставлена 3D-модель и определены опорные точки.
5. **3D temporal tracking:** объединять измерения во времени, оценивать скорость, фильтровать шум, ограниченно прогнозировать короткие пропуски.
6. **Evaluation/export:** сравнивать с Ground Truth на тестовом наборе и выдавать траекторию в формате, заданном организаторами.

## 1.3. Что в презентации Беспальцева релевантно

Файл «Беспальцев А..pptx» шире конкурсного задания и включает две большие темы:

- **Раздел 01 — визуальная навигация без GNSS.** Это общий профессиональный контекст компании; по текущему ТЗ грузовика это не основная конкурсная задача. Не нужно добавлять в PecStorm глобальную локализацию по спутниковым картам или визуальную одометрию, если отдельное требование на это не появится.
- **Раздел 02 — захват и сопровождение объектов.** Это непосредственно релевантно. На слайдах 10–14 перечислены сложности: маленький/низкоконтрастный объект, изменение масштаба и ракурса, перекрытия/выход из кадра, ограниченные вычисления, оценка уверенности и повторный захват. Рассматриваются классические подходы, OSTrack, MixFormer, SAMURAI, MaST, а также метрики tracking benchmark.

Упоминание MaST/OSTrack/MixFormer в презентации — исследовательский контекст, а не обязательное требование использовать именно эти модели. Выбирать модель нужно по тестам на грузовике, задержке и доступному GPU.

---

# 2. ВХОДНЫЕ МАТЕРИАЛЫ И ЧТО В НИХ НАХОДИТСЯ

В пакет переданы:

1. `project2(1).7z` — архив с исходным заданием/данными; содержит пять видеозаписей и шесть CSV.
2. `Зуенок А..pptx` — презентация с формулировкой ТЗ, базовой схемой, структурой входных данных, графиком сдачи и примером результата.
3. `Беспальцев А..pptx` — техническая презентация о визуальном интеллекте, особенно разделе захвата и сопровождения.
4. `PecStorm-main.zip` — текущий репозиторий команды.
5. `README.md` — описание состояния текущей реализации.

В handoff package рядом с этим документом должны лежать оригинальные вложения; см. `source_materials/`.

## 2.1. Пять видео

Названия (по составу архива):

- `Way1.FinalImage.CineCameraActor1.mp4`
- `Way1.FinalImage.CineCameraActor2.mp4`
- `Way1.FinalImage.CineCameraActor3.mp4`
- `Way1.FinalImage.CineCameraActor4.mp4`
- `Way1.FinalImage.CineCameraActor5.mp4`

Проверенные/визуально извлечённые параметры видео:

- разрешение: 1920 × 1080;
- частота: 30 FPS;
- число декодируемых кадров: около 1001 (индексы 0–1000);
- длительность: около 33,37 секунды;
- кодек: H.264;
- сцена: синтетическая 3D-сцена с грузовиком на заснеженной горной дороге.

Перед кодированием этих значений в программу повторно проверь их через `ffprobe`/OpenCV на каждом файле. Не предполагай, что будущий проверочный набор будет иметь те же свойства.

## 2.2. Пять CSV с параметрами камер

Файлы:

- `object_transform_CineCameraActor1.csv`
- `object_transform_CineCameraActor2.csv`
- `object_transform_CineCameraActor3.csv`
- `object_transform_CineCameraActor4.csv`
- `object_transform_CineCameraActor5.csv`

Это **не** CSV с 2D-детекциями. В них покадровые/постоянные параметры камер:

`Frame; Loc_X; Loc_Y; Loc_Z; Rot_Roll; Rot_Pitch; Rot_Yaw; ...; Focal length`

Файлы имеют 1002 записи с `Frame=0..1001`. В конце первых строк также встречаются метаданные сенсора `23,76*13,365мм` и фокусное расстояние в заголовке/дополнительном поле. Это не аккуратный машинный CSV: там есть дополнительные пустые колонки и строка с русским текстом.

### Параметры камер, прочитанные из CSV

Координаты и углы ниже приведены в исходных единицах таблицы. Вероятно, `Loc` заданы в сантиметрах Unreal Engine; единицы нужно подтвердить.

| Камера | Loc_X | Loc_Y | Loc_Z | Roll | Pitch | Yaw в начале | Focal length |
|---|---:|---:|---:|---:|---:|---:|---:|
| CineCameraActor1 | -24701.9592 | 6623.5717 | 1200.0 | 0.0° | 0.0° | -70.0° | 35 mm |
| CineCameraActor2 | -45784.0013 | -6500.0 | 7911.8821 | 0.0° | -10.0° | 0.0° | 60 mm |
| CineCameraActor3 | 12600.0 | -22700.0 | 1400.0 | -0.6776° | -3.114° | 115.1096° | 35 mm |
| CineCameraActor4 | 33281.733 | 13996.6567 | 7146.7354 | 0.0° | -10.0° | -160.0° | 50 mm |
| CineCameraActor5 | 9799.8057 | -3900.0 | 1487.5303 | -4.4117° | -6.9741° | -164.0° | 20 mm |

Наблюдения по параметрам:

- Камеры 1–4 имеют постоянные `Loc` и `Rot` на всех строках CSV.
- Камера 5 имеет постоянный `Loc`, `Roll`, `Pitch`, но меняющийся `Rot_Yaw`.
- `Rot_Yaw` камеры 5 проходит через границу `+180/-180°`: сырые значения лежат примерно от `-179.8084°` до `+179.7232°`; старт `-164°`, к концу `104°`.
- При непрерывном unwrap это соответствует повороту примерно от `-164°` до `-256°` (около -92°), а не резкому скачку на 360°.

**Критическое следствие:** нельзя использовать одну неизменную внешнюю позу для камеры 5. Её ориентация должна обновляться по Frame. Не интерполировать сырые углы линейно через ±180°; лучше переводить ротации в матрицы/quaternions и при необходимости применять SLERP.

## 2.3. Предварительные внутренние параметры камер

В CSV присутствует размер сенсора `23.76 × 13.365 mm`, соотношение 16:9. При разрешении 1920×1080 и при допущениях, что:

- focal length указан в миллиметрах;
- используется весь сенсор без crop/overscan;
- principal point находится ровно в центре;
- нет неизвестного изменения sensor/filmback;

можно получить первичную оценку:

`fx = focal_mm * 1920 / 23.76`

`fy = focal_mm * 1080 / 13.365`

`cx = 960`, `cy = 540`.

| Камера | fx ≈ fy, px | Приблизительный горизонтальный FOV |
|---|---:|---:|
| 1 (35 mm) | 2828.28 | 37.50° |
| 2 (60 mm) | 4848.48 | 22.40° |
| 3 (35 mm) | 2828.28 | 37.50° |
| 4 (50 mm) | 4040.40 | 26.73° |
| 5 (20 mm) | 1616.16 | 61.42° |

**Это стартовая оценка intrinsics, не подтверждённая калибровка.** Уточнить у организаторов точный filmback/sensor fit, principal point, crop, distortion и то, экспортированы ли camera transform значения до или после lens shift. В программе явно хранить provenance/assumptions. Не выдавать нулевую дисторсию за измеренный факт.

## 2.4. Шестой CSV — Ground Truth грузовика

Файл:

`object_transform_SM_AA_track.csv`

Колонки:

`Frame; Loc_X; Loc_Y; Loc_Z`

Это эталонная 3D-траектория/transform объекта SM_AA_track, а не детекции. В нём 1002 строки, `Frame=0..1001`. Видео визуально содержит около 1001 кадров, индексы 0–1000; последнюю строку нужно проверить на дублирование и не обрабатывать кадр 1001 как существующий видеокадр без подтверждения.

Диапазоны числовых значений (две повреждённые текстовые записи исключены из numeric min/max):

- X: примерно от -21500 до 11513.2402;
- Y: примерно от -12800 до 11105.5596;
- Z: от 300 до 717.9231.

Данные очень похожи на Unreal world coordinates в сантиметрах. В таком случае в метрах координаты будут `xyz_cm / 100`. Но единицы и точка отсчёта должны быть подтверждены. Не перемешивать сантиметры в CSV с метрами, которыми оперирует текущая библиотека.

### Повреждённые значения в Ground Truth CSV

Из-за форматирования/автоконвертации Excel в две числовые ячейки попал текст с названиями месяцев (файл в кодировке CP1251):

- `Frame=402`, `Loc_X` записано как `авг.13`;
- `Frame=691`, `Loc_Y` записано как `июл.25`.

По соседним точкам траектории эти значения, вероятно, были десятичными числами `8.13` и `7.25` соответственно, которые Excel интерпретировал как даты (August 13 / July 25). Это **гипотеза на основе локали и непрерывности соседей**, а не гарантированно подтверждённые исходные значения.

Правильное поведение:

1. Не изменять оригинальный CSV.
2. Создать `data/truck/ground_truth_repairs.json` с явными override для `Frame=402/Loc_X` и `Frame=691/Loc_Y`, если команда принимает восстановленные значения.
3. Сохранить запись `source_raw_value`, `proposed_value`, `reason`, `verified_by_organizer=false`.
4. Поддержать режим `strict`, который останавливает evaluation при повреждённом GT, пока override явно не разрешён.
5. Запросить у организаторов исходную таблицу без Excel date coercion. Не утверждать MAE/RMSE по GT до решения этих аномалий.

## 2.5. Визуально проверенная доступность объекта по камерам

По контрольным contact sheets/кадрам (выборочные проверки, не полная покадровая разметка) наблюдалось:

| Время около | Камеры, где грузовик заметен на выборочном кадре |
|---:|---|
| 0 s | 1 |
| 5 s | 1, 2 |
| 10 s | 2 |
| 15 s | 2, 3, 5 |
| 20 s | 3, 5 |
| 25 s | 4, 5 |
| 30 s | 4, 5 |

Это иллюстрирует смену зон видимости и участки с единственным ракурсом. Не считать таблицу точной visibility ground truth: реализовать автоматическую/ручную оценку для всего видео и хранить маску видимости отдельно.

**Ключевая архитектурная проблема:** классическая триангуляция возможна, когда есть как минимум два корректных соответствующих 2D-наблюдения одного и того же физического 3D-якоря в один момент времени. На участках, где виден только один ракурс, глубина по одной 2D-точке принципиально не определяется без дополнительной информации (3D-модель, keypoint correspondences, ground/road constraints, known object dimensions, motion prior и т.п.). Kalman может кратковременно прогнозировать положение, но это прогноз, а не новое измерение.

## 2.6. 3D-модель грузовика

В слайде 9 презентации ТЗ 3D-модель объекта перечислена как часть структуры проекта. В переданном архиве `.7z` по доступному списку материалов обнаружены 5 видео и 6 CSV; отдельный файл 3D-модели не обнаружен.

Нужно запросить модель и уточнить формат (FBX/OBJ/GLTF/Unreal asset), pivot/origin, масштаб, оси и соответствует ли `SM_AA_track` actor origin этой модели.

Если модель будет предоставлена, она может дать возможность model-based monocular localization / PnP по 2D–3D keypoints/feature points. Без модели или надёжных известных опорных точек нельзя обещать метрическую глубину по единственной камере.

---

# 3. ЧТО УЖЕ ЕСТЬ В PECSTORM

Текущий архив `PecStorm-main.zip` содержит Python-библиотеку `multiview3d`, CLI `apps/`, тесты, конфигурацию, `README.md`, `HANDOFF.md`.

Основные модули:

| Файл | Текущая роль | Действие для задания Пеленга |
|---|---|---|
| `multiview3d/types.py` | `CameraCalibration`, `Detection2D`, `Observation3D`, `TrackPoint`, `Track` | Добавить/расширить поля статуса и качества, если нужно, сохраняя совместимость |
| `camera.py` | проекция, undistortion, back-projection ray, DLT/лучевая триангуляция, refinement, reprojection, fundamental matrix | Переиспользовать; тщательно проверить конверсию Unreal/OpenCV и динамические camera states |
| `calibration.py` | калибровка шахматной доской/кругами, PnP, bundle adjustment, сохранение/загрузка | Добавить importer данных Unreal/CSV отдельно от процедуры шахматной калибровки |
| `detect.py` | `Detector`, `YoloDetector`, `OnnxDetector`, `ColorBlobDetector`, `OracleDetector` | Настроить детекцию truck; проверить реальные ML backend |
| `capture.py` | `FrameSource`, `OpenCvFrameSource`, `FrameReader`, latest buffer, frame alignment/offline readers | Исправить интеграцию detector для file/stream, синхронизацию и EOF поведения |
| `fusion.py` | эпиполярное сопоставление, Hungarian, группировка видов, триангуляция, MultiViewTracker | Поддержать динамические позы камер по кадру и метаданные качества наблюдения |
| `kalman.py` | 6D constant-velocity Kalman, Mahalanobis gate, RTS smoothing | Проверить gaps, adaptive measurement noise, online/offline semantics |
| `trajectory.py` | признаки, smoothing, resampling, match-to-truth | Использовать для оценки, не путать интерполированное GT и measurement |
| `sim.py` | синтетические камеры/объекты, рендер, GT, SimSource | Сохранить для unit/stress tests; truck dataset сделать отдельным data adapter |
| `pipeline.py` | PipelineConfig, build_sources, run/run_offline, step, history, RTS | Добавить режим/адаптер truck files и per-frame camera states |
| `metrics.py` | 3D error, completeness/track metrics/FPS | Дополнить coverage, validity masks, missing/reacquisition, online-vs-offline и latency |
| `visualize.py` | 3D trajectory plots, overlays, CSV/JSON export | Добавить truck diagnostic dashboard/overlay |
| `apps/sim_dataset.py` | синтетическое демо | Сохранить |
| `apps/evaluate.py` | оценка tracks vs truth | Расширить/настроить для truck GT и валидной маски кадров |
| `apps/calibrate.py` | calibration workflow | Не использовать как substitute Unreal importer |
| `apps/live.py` | console live, экспорт треков | Сделать offline video evaluation отдельным CLI и real-time отдельно |

## 3.1. Текущая геометрическая конвенция CameraCalibration

В `multiview3d/types.py` документация задаёт:

`X_cam = R @ X_world + t`

`projection_matrix = K @ [R | t]`

То есть `R,t` — **world-to-camera** extrinsics. Центр камеры в мире:

`C_world = -R.T @ t`

В `fusion.py:cam_center()` это формула уже применяется.

В `CameraCalibration` поле `unit` документировано как multiplier, но текущие функции `project_point`, `back_project_ray`, `cam_center` используют непосредственно `R,t` и не выполняют автоматическое масштабирование координат через `unit`. Поэтому нельзя просто записать сантиметровые позиции/translation и надеяться, что `unit=0.01` всё исправит. Приведи `X_world`, camera centers и `t` к одной системе единиц до создания `CameraCalibration`.

Текущая библиотека описывает 3D-позиции объектов в метрах. Рекомендуемый внутренний стандарт: **метры**. Преобразование Unreal centimeters → meters сделать один раз на входе в adapter. Для world-to-camera:

1. Получить ориентацию камеры и её центр `C_world` в общей мировой системе.
2. Построить корректную `R_world_to_camera` в OpenCV convention.
3. Вычислить `t = -R @ C_world`.
4. Убедиться, что `C_world` и `t` в метрах.

Не смешивать yaw/pitch/roll convention UE с OpenCV без явного преобразования осей/порядка вращений.

## 3.2. Baseline тестов

По текущему репозиторию было выполнено:

`python -m unittest discover -s tests -v`

**23 теста прошли (OK).** Это подтверждает текущие unit/synthetic tests, но не проверяет truck dataset или реальные YOLO/ONNX detectors.

README описывает synthetic baseline: 3 камеры, 2 объекта, 25 FPS; один прогон с Oracle имеет среднюю ошибку около 10.14 мм, другой с Blob — около 7.48 мм, ID-switch 0. В README также указано около 260 FPS для синтетического демо. Отдельный запуск того же проекта в другой среде/условиях дал другие значения производительности; не использовать эти цифры как KPI для truck dataset. Сравнивать измерения только в одинаковом режиме, на одинаковом hardware и с чётко определённым таймером.

---

# 4. ИЗВЕСТНЫЕ ПРОБЛЕМЫ И ТЕХНИЧЕСКИЕ РИСКИ

Ниже проблемы, которые были обнаружены при чтении исходников/проверках текущего архива. Coding agent должен подтвердить их на рабочей копии и оформить тестами.

## P0 — реальная ветка видео не запускает detector

В `pipeline.py`, `MultiViewPipeline.build_sources()`:

- ветка `kind == "sim"` создаёт `SimSource(..., build_detector(spec))`;
- ветка `kind in ("file", "stream")` создаёт только `OpenCvFrameSource(sc.source)`.

`OpenCvFrameSource.read()` в `capture.py` возвращает `(stamp, frame, None)`. Затем `_detections_from_aligned()` превращает `None` в пустой список. Поэтому конфиг YOLO/Blob не запускается на MP4/RTSP в текущем file/stream path.

**Требование:** добавить общий detector adapter/source decorator, например `DetectingFrameSource`, который запускает detector на каждом кадре и выдаёт detections в одном FrameSource контракте. Поддержать per-stream detector и глобальный default. Написать интеграционный тест на видеофайле с известным объектом.

## P0 — текущий pipeline статичен по камерам

`MultiViewPipeline` строит `MultiViewTracker(self.names, self.cameras, ...)` при инициализации. `fusion.build_correspondences()` и `triangulate_group()` получают постоянный список камер. Это не поддерживает переменную ориентацию CameraActor5.

**Требование:** добавить явный API для camera state на каждый Frame, например `step(timestamp, detections_by_name, cameras=None)` или отдельный `CameraStateProvider`. Для каждого frame использовать `CameraCalibration`/pose snapshot, не мутировать глобальные матрицы в unsafe shared-thread context. Синтетические/static callers должны остаться совместимыми. Тест: динамическая камера с известной позой и synthetic projections восстанавливает GT.

## P0 — `read_aligned()` требует кадр от каждой камеры

`capture.read_aligned()` возвращает `None`, если `len(frames) < len(readers)` и если число aligned views меньше общего числа readers. Это блокирует обработку при потере одной камеры даже если `FusionConfig.min_views=2` и две другие камеры доступны.

**Требование:** различать camera stream availability и per-frame detections; разрешить неполный набор камер при наличии минимум `min_views` камер с валидными наблюдениями. Не использовать отсутствующий frame как синхронное пустое наблюдение без явной политики времени. Добавить camera dropout test.

## P0/P1 — буфер называется LatestFrameBuffer, но возвращает старейший из оставшихся кадров

`LatestFrameBuffer(maxlen=2)` вытесняет старое при заполнении, но `get()` возвращает элемент очереди FIFO. Это может вернуть более старый из двух кадрів. Для latest-frame policy нужно атомарно забирать самый новый элемент и удалять старые, либо хранить одиночный latest slot. Для offline processing использовать очередь без пропуска кадров.

## P0/P1 — video timestamp сейчас вычисляется как index / fps

`OpenCvFrameSource.read()` использует `stamp = self._index / self.fps`. Это приемлемо для синхронных видеофайлов с одинаковой частотой, общим началом и отсутствием пропущенных/переменных timestamps. Это не аппаратный timestamp для реальной камеры. Для предоставленного synthetic video dataset можно начать с frame index alignment; для неизвестного набора надо проверять PTS/actual fps и frame offsets.

## P1 — EOF/остановка FrameReader

Проверить поведение `OpenCvFrameSource` на EOF: `read()` возвращает `None`, но `closed` основан на `VideoCapture.isOpened()`, который может остаться true до release. Тогда live run может ждать stall timeout. Добавить `eof`/`is_exhausted` semantics или явно release on EOF; не ломать RTSP reconnect behavior.

## P1 — timestamps разных камер должны быть действительно согласованы

`frame_index / fps` для всех камер нулит старт каждого видео. Если в реальности видео с разных камер стартовали не одновременно, frame index не означает общий физический момент. Для текущего synthetic dataset это, вероятно, общий timeline, но это необходимо подтвердить. Добавить frame offsets / per-camera PTS handling.

## P1 — точка детекции

Текущий `Detection2D.center` обычно центр bbox. Межкамерная триангуляция корректна только если два наблюдения соответствуют одной физической 3D-точке. Центры bbox меняются по перспективе и ракурсу и не обязаны быть проекциями одного 3D-якоря.

**Требование:** отделить bbox center от triangulation keypoint. Поддержать `keypoints`/anchors и их confidence. Для объекта по имени класса truck проверить, достаточно ли одного устойчивого keypoint. Если точка Ground Truth — actor pivot, для неё нужна привязка к модели/геометрии, а не произвольный центр bbox.

## P1 — калибровка может быть неверной при неверной общей позе мишени

Текущий код `calibrate_camera_from_images()` использует несколько изображений для intrinsics, а одна поза первой доски/кадра используется в `solvePnP` для extrinsics. Для отдельных независимых наборов шахматной доски это не даёт автоматически общей world frame между камерами. Для truck dataset не использовать этот calibration flow вместо CSV importer.

## P1 — FPS метрика не end-to-end

В `pipeline.step()` время замеряется вокруг `tracker.step()`. Это не включает capture, decode, detector, synchronization и export/render. Добавить per-stage timings и end-to-end latency.

## P1 — live app печатает/экспортирует результат после завершения

`apps/live.py` вызывает `pipe.run()`, а затем показывает сглаженные треки. Для настоящий streaming UI/output нужен per-step callback/iterator/queue. Для contest dataset может быть допустим offline exporter, но не называть это online live output.

## P1 — real-time фильтр vs RTS

Kalman filter — online. RTS smoothing использует будущие наблюдения и является offline/fixed-lag методом. Отдельно маркировать `online_state`, `offline_smoothed_state`, `predicted_only`.

## P2 — геометрическая обусловленность и uncertainty

Сейчас есть reprojection error; добавить оценку угла между лучами, conditioning/covariance. Маленький reprojection error сам по себе не доказывает малую 3D-ошибку, особенно по глубине при плохом baseline.

---

# 5. ЦЕЛЕВАЯ АРХИТЕКТУРА ДЛЯ ЗАДАНИЯ ПЕЛЕНГА

Не ломать существующий `multiview3d`. Добавить adapter layer и отдельный truck workflow.

Предлагаемая схема:

```text
                  DATASET / VIDEO INPUT
       ┌──────────┬──────────┬──────────┬──────────┬──────────┐
       │ cam1.mp4 │ cam2.mp4 │ cam3.mp4 │ cam4.mp4 │ cam5.mp4 │
       └────┬─────┴────┬─────┴────┬─────┴────┬─────┴────┬─────┘
            │          │          │          │          │
       Frame/PTS alignment + per-camera metadata
            │          │          │          │          │
       ┌────▼──────────▼──────────▼──────────▼──────────▼────┐
       │ Camera provider: static poses 1-4, dynamic pose 5   │
       │ Unreal CSV -> OpenCV CameraCalibration per frame    │
       └──────────────────────────┬───────────────────────────┘
                                  │
       ┌──────────────────────────▼───────────────────────────┐
       │ Per-camera object detection                          │
       │ YOLO baseline / optional keypoint or segmentation    │
       └──────────────────────────┬───────────────────────────┘
                                  │
       ┌──────────────────────────▼───────────────────────────┐
       │ Per-camera tracker: TRACKING / LOST / REACQUIRE      │
       └──────────────────────────┬───────────────────────────┘
                                  │
       ┌──────────────────────────▼───────────────────────────┐
       │ 2D target observations with same anchor/keypoints    │
       └──────────────────────────┬───────────────────────────┘
                                  │
                ┌─────────────────┴──────────────────┐
                │                                    │
       >= 2 valid camera views               1 valid camera view
                │                                    │
       Multi-view association                 Model-based monocular
       Epipolar gating                        PnP / 3D keypoints / shape
       Triangulation + refinement              ONLY if 3D model/anchors exist
                │                                    │
                └─────────────────┬──────────────────┘
                                  │
                       Measurement + quality
                                  │
                       3D temporal state estimator
                  Kalman / adaptive covariance / gap handling
                                  │
                       online estimate + status
                                  │
                 optional offline RTS/fixed-lag smoother
                                  │
                     CSV/JSON trajectory + evaluation
```

## 5.1. Предлагаемая структура новых файлов

Сначала реализовать минимальный набор. Не создавать все файлы без необходимости.

```text
PecStorm/
├── multiview3d/
│   ├── ... existing modules ...
│   ├── sources.py                 # DetectingFrameSource / source adapters (если уместно)
│   ├── camera_provider.py         # camera state per frame
│   ├── unreal_camera.py           # Unreal CSV and coordinate conversion
│   ├── target_tracking.py         # target state, confidence, lost/reacquire (или в detect.py)
│   └── uncertainty.py             # 3D measurement covariance (P2)
│
├── datasets/
│   └── truck/                     # оригинальные входы не модифицировать
│       ├── videos/
│       ├── camera_params/
│       ├── ground_truth.csv
│       ├── ground_truth_repairs.json
│       └── dataset_manifest.json
│
├── apps/
│   ├── import_truck_dataset.py
│   ├── detect_truck.py
│   ├── reconstruct_truck.py
│   ├── evaluate_truck.py
│   └── visualize_truck.py
│
├── experiments/
│   ├── visibility_analysis.py
│   ├── projection_sanity.py
│   ├── detector_benchmark.py
│   ├── geometry_sweep.py
│   └── robustness_benchmark.py
│
├── tests/
│   ├── test_unreal_csv.py
│   ├── test_coordinate_conversion.py
│   ├── test_dynamic_camera.py
│   ├── test_video_detector_source.py
│   ├── test_partial_camera_dropout.py
│   ├── test_projection_to_video.py
│   ├── test_truck_gt_loader.py
│   └── test_truck_evaluation.py
│
└── docs/
    ├── TRUCK_DATASET.md
    ├── OPEN_QUESTIONS.md
    └── RESULTS.md
```

Не обязательно сохранять ровно такую структуру, если существующие соглашения проекта подсказывают лучшее место. Важно изолировать входной формат Unreal от общей геометрии.

---

# 6. ПЛАН РАБОТ ПО ПРИОРИТЕТАМ

## P0 — блокирующие задачи; делать первыми

### P0.1. Dataset loader и manifest

- Найти/распаковать пять MP4 из `project2(1).7z`.
- Проверить `ffprobe`: размеры, fps, time base, frame count, PTS, duration, codec.
- Прочитать camera CSV с `sep=';'`, правильной encoding (исходные CSV по факту читаются как CP1251/Latin-1, текст метаданных на русском).
- Не использовать `sep=None` autodetect без теста; заголовок содержит дополнительные поля/пустые колонки/metadata.
- Читать только нужные numeric columns.
- Сохранять camera Frame 0..1001.
- Прочитать GT Frame 0..1001, обработать invalid values по описанной strict/override политике.
- Сформировать manifest с mapping `camera_id ↔ video ↔ camera_csv`, fps, resolution, frame ranges, units assumptions, source file hashes.

**Acceptance:** одна команда проверяет целостность набора и выдаёт понятный отчёт без молчаливых coercion/NaN.

### P0.2. Camera coordinate adapter

- Выяснить Unreal coordinate convention (оси, handedness, порядок Euler rotations, единицы, camera local axes).
- Построить `K` для каждой камеры с explicit assumptions.
- Перевести world camera centers из исходных единиц в метры.
- Построить OpenCV world-to-camera `R,t`, учитывая разницу coordinate frames.
- Для cam1–4 вернуть static calibration.
- Для cam5 получить per-frame calibration/pose; учитывать yaw wrap через quaternion/rotation matrices.
- Учитывать возможную смену ориентации в каждом кадре.
- Сохранять raw Unreal transform и converted OpenCV pose для диагностики.

**Acceptance:** `project_point()`/`project_points()` для тестовой 3D-точки выдаёт пиксель в правильной области конкретного кадра с учётом видимой геометрии. Пока нет соответствующих keypoints/3D модели, не требовать пиксельного совпадения с actor origin; использовать bbox sanity checks и отдельно запросить anchor definition.

### P0.3. Detector должен работать с file sources

- Исправить `MultiViewPipeline.build_sources()` так, чтобы `file`/`stream` sources действительно пропускали кадры через detector.
- Добавить decorator/adapter, который возвращает `(timestamp, frame, detections)`.
- Поддержать global detector и detector per stream.
- Проверить YOLO на пяти видео (сначала готовая модель с классом truck).
- Сохранять detection CSV/JSON, overlays и score.

**Acceptance:** один CLI прогон каждого MP4 создаёт валидные 2D detections; нет пустого списка лишь из-за неправильной интеграции.

### P0.4. 2D point / anchor semantics

- Сначала использовать bbox для простого detection baseline.
- Оценить bbox center, нижний центр, center of visible mask, keypoints.
- Не выбирать метод триангуляции до определения 3D anchor в `SM_AA_track`.
- Если 3D-модель недоступна, запросить ее; временно сравнить несколько потенциальных 2D anchors и визуально проверить их consistency.

**Acceptance:** документировано, какая конкретно 2D-точка на каждом ракурсе соответствует целевой 3D-точке. Если это пока неизвестно, результат маркируется как provisional.

### P0.5. Per-frame camera states в fusion

- Передавать актуальный набор camera states на каждом `pipeline.step`.
- Сделать camera provider, который по Frame выдаёт calibrations для 5 камер.
- Синхронизировать время детекций и camera transform по Frame/PTS.
- Проверить поворот Camera5 на кадрах после начала вращения.

**Acceptance:** synthetic test с динамической камерой восстанавливает заданную точку в 3D с ожидаемой точностью.

### P0.6. Полноценная Truck Evaluation

- Оценивать только кадры, где есть валидный Ground Truth и действительно получена оценка.
- Считать coverage отдельно от error.
- Делить метрики по методам: triangulation, monocular/model-based, predicted-only.
- Не смешивать GT-repairs с исходными данными; выводить флаг.

**Acceptance:** `evaluate_truck.py` создаёт CSV/JSON summary и графики ошибки по frame/time/camera subset/method.

## P1 — надёжность, сопровождение и качество

### P1.1. Partial camera availability

- Исправить `read_aligned()` / orchestration: отсутствие одной камеры не должно останавливать систему, если доступно достаточно синхронных наблюдений.
- Учесть разные кадры/PTS; не соединять устаревшие кадры с новыми без проверки временного допуска.
- Для кадра без двух views выдать `UNOBSERVABLE`/monocular result/predicted-only, а не fake triangulation.

### P1.2. Per-camera target tracker

Сделать независимый локальный трекер грузовика на каждой камере с понятными состояниями:

- `UNINITIALIZED`
- `TRACKING`
- `LOW_CONFIDENCE`
- `LOST`
- `REACQUIRING`

Нужны confidence gating, расширение области поиска после потери, повторная детекция и защита от переключения на похожий объект.

### P1.3. Model-based monocular position (только при наличии данных)

Если организаторы предоставляют 3D truck model и известен pivot/anchors:

- выбрать 3D keypoints модели;
- обучить/получать 2D keypoints или matching features;
- применить `solvePnP`/robust PnP с outlier rejection;
- определить scale/unit и pose convention;
- оценить reprojection residual и pose covariance;
- тестировать на synthetic projection of truck model.

Если модели нет, не симулировать абсолютную глубину по одной камере. Оставить interface и status `PREDICTED_ONLY`.

### P1.4. Kalman / gaps / smoothing

- проверить модель CV на реальном GT грузовика;
- оценить динамические ускорения и turn rates;
- не считать один `max_missing` в кадрах без связи с fps/time;
- использовать корректный `dt` из timestamps;
- при необходимости adapt measurement covariance по reprojection error/triangulation angle/detector confidence;
- online estimate и offline RTS должны быть раздельными output products.

### P1.5. Video processing semantics

- В offline режиме не пропускать кадры из-за latest-frame queue.
- В live режиме latest-frame policy должна реально выдавать самый новый кадр.
- Явно распознавать EOF для файлов.
- Отдельно обрабатывать reconnect для RTSP/камер.
- Замерять capture/decode/detection/association/reconstruction/tracking/export и end-to-end latency.

## P2 — улучшения после работающего baseline

- Угол пересечения лучей/triangulation conditioning.
- Covariance / uncertainty 3D measurement.
- Confidence-weighted triangulation / robust outlier rejection.
- Geometry sweeps и оценка вклада камер.
- Tracking benchmark: loss rate, reacquisition delay, stability.
- Реалтайм dashboard и 3D visualization.
- ONNX/GPU acceleration; сравнить speed/accuracy.
- Fine-tuning detector на вручную размеченных кадрах — только при подтверждённом провале pretrained model.

---

# 7. ТЕСТОВЫЙ ПЛАН И КРИТЕРИИ ПРИЁМКИ

## 7.1. Обязательные тесты данных и камеры

1. `test_camera_csv_parse`: корректно считывать semicolon/CP1251, metadata rows, frame numbers.
2. `test_gt_csv_invalid_cells`: `авг.13` и `июл.25` обнаруживаются как ошибки, а не silently converted.
3. `test_gt_override_audit`: sidecar repair явно маркируется и не меняет исходный файл.
4. `test_camera_units`: world coordinates and translation have same units; check meters.
5. `test_unreal_to_opencv_axes`: известная точка/направление переводится правильно.
6. `test_static_camera_pose`: cameras 1–4 remain stable across frames.
7. `test_camera5_yaw_unwrap`: no artificial 360° jump; rotation matrix remains continuous.
8. `test_video_frame_alignment`: одинаковые frame indices correspond to synchronized timeline only if PTS confirms.
9. `test_frame_count_mismatch`: обработка общей области кадров 0..1000; extra row 1001 не вызывает off-by-one.

## 7.2. Projection sanity tests

1. Прочитать Ground Truth frame f (только для теста).
2. Проецировать actor location через камеру f.
3. Сверить: попадает ли projected actor anchor в экран и в видимую область/окрестность грузовика.
4. Повторить для нескольких камер/кадров, включая Camera5 во время поворота.
5. Если точка не попадает около грузовика — отладить coordinate transform, axes, Euler order, camera local frame, units, image origin, focal/principal point.

ВНИМАНИЕ: actor pivot не обязательно совпадает с 2D bbox center. До выдачи 3D-модели/keypoint anchor projection sanity здесь лишь проверяет грубую согласованность, а не точность пиксельного трекинга.

## 7.3. Detection tests

- вручную разметить небольшой stratified subset: разные камеры, временные точки, масштабы, дальние/близкие планы, частичные occlusions;
- измерить precision/recall, IoU, center/keypoint error;
- отдельный отчёт per camera;
- оценить confidence calibration;
- сохранить overlays с false positives/false negatives;
- сначала попробовать pretrained `truck` class;
- если грузовик в кадре слишком мал/низкоконтрастен, рассмотреть fine-tune detector на кадрах из train split, не на финальном hidden test.

## 7.4. Reconstruction tests

- парные камеры 1–2, 1–3 и т. д. на интервалах совместной видимости;
- все доступные камеры на каждом кадре;
- dynamic Camera5;
- synthetic 3D projections через ту же camera model;
- шум 2D keypoints (0, 0.25, 0.5, 1, 2, 5 px);
- calibration perturbation (intrinsics/extrinsics);
- timestamp offsets (1, 2, 5, 10, 20, 50 ms), если применимо;
- reject ill-conditioned triangulations;
- проверить что reprojection residual мал, но не считать его эквивалентом GT 3D error.

## 7.5. Tracking tests

- непрерывная видимость;
- один объект пропадает в одной камере;
- объект пропадает во всех камерах на короткий промежуток;
- выход/повторный вход в поле зрения;
- смена масштаба;
- перекрытие деревьями/рельефом;
- ложные detections;
- ID switch / reacquisition.

## 7.6. Метрики

### 3D position accuracy

`e_i = ||p_pred_i - p_gt_i||_2`

Считать:
- MAE 3D;
- RMSE 3D;
- median/P50;
- P95;
- P99;
- max error;
- per-axis MAE/RMSE;
- error vs time/frame;
- error per camera combination/method.

### Coverage / observability

- доля кадров с валидным 3D measurement;
- доля кадров с monocular pose estimate;
- доля кадров predicted-only;
- доля кадров без оценки;
- visibility / number of contributing cameras.

Нельзя сообщать только MAE на успешно восстановленных кадрах: нужно показать coverage, иначе алгоритм может выглядеть точным, восстановив лишь лёгкие интервалы.

### 2D / tracking

- precision / recall / F1 detections;
- IoU / center error / keypoint error;
- loss rate;
- reacquisition time;
- fragmentation;
- ID-switch, если есть несколько tracks/hypotheses.

### Runtime

- per-stage timings;
- end-to-end latency;
- processing FPS;
- source FPS;
- dropped frames;
- peak memory / device utilization, если доступно.

Отделять online metrics и offline smoothed metrics.

## 7.7. Главный интеграционный Acceptance Test

**Цель:** пять видео → CSV cameras → детекция/сопровождение → доступные 3D наблюдения → оценки → экспорт.

Успешный тест должен:

1. загрузить пять видео и пять camera CSV;
2. привести frame indices/timestamps к согласованной шкале;
3. построить static poses для камер 1–4 и per-frame pose для камеры 5;
4. запустить детектор на каждом кадре каждого MP4;
5. сохранить 2D detections и overlays;
6. выполнить reconstruction на кадрах с >=2 valid views;
7. корректно обработать кадры с одним view как monocular/predicted/unobservable;
8. записать 3D estimates с frame/time, method, status, confidence/quality и contributing cameras;
9. сравнить valid measured estimates с Ground Truth после явного разрешения аномальных GT cells;
10. сформировать summary JSON, CSV и графики.

Не требовать заранее выдуманной численной точности. Сначала получить baseline и выяснить официальную метрику/допуск у организаторов. Если организаторы сообщат numeric target, закрепить его в acceptance config.

---

# 8. ПРЕДЛАГАЕМАЯ РАБОТА С НЕЙРОСЕТЯМИ

## 8.1. Детектор

Начальный baseline: готовый YOLO с классом `truck` (проверить конкретную версию/weights и лицензирование). Не обучать с нуля.

Проверить кадры всех камер. Для small/distant truck стандартный COCO detector может пропускать объект; тогда:

1. сделать ручную разметку выборки;
2. измерить baseline;
3. дообучить готовую модель при достаточном объёме данных;
4. отдельно проверить hidden/verification data, не переобучаясь под один тестовый набор.

## 8.2. Tracker

ТЗ — один заданный объект, поэтому разумно оценить:

- YOLO detection + ByteTrack/BoT-SORT как простой robust baseline;
- single-object tracker (например, OSTrack/MixFormer/SAMURAI или другой доступный backend) после инициализации;
- detector-assisted reacquisition после `LOST`.

Не требуется внедрять все алгоритмы. Выбрать лучший по tracking accuracy, reacquisition, latency и стабильности на данном датасете.

## 8.3. Monocular 3D

В презентации среди входных данных указана 3D-модель. Если она будет предоставлена и её pivot/размеры соответствуют target transform:

- получить набор 3D model keypoints;
- определить соответствующие 2D keypoints на грузовике;
- решать PnP (например, RANSAC PnP) с проверкой reprojection;
- получить положение/ориентацию модели и нужный target anchor;
- оценивать uncertainty и condition.

Если модели нет, нельзя получить однозначную metric depth только из одного bounding box/одной 2D точки. Возможные дополнительные ограничения (дорога/высота/размер/движение) должны быть явно подтверждены и проверены, а не выдуманы.

---

# 9. ВОПРОСЫ ОРГАНИЗАТОРАМ — БЛОКИРУЮЩИЕ НЕИЗВЕСТНЫЕ

Создать `docs/OPEN_QUESTIONS.md` и зафиксировать ответы.

1. Где 3D-модель грузовика, указанная на слайде 9? В каком формате?
2. Какую именно точку нужно возвращать: actor pivot/origin `SM_AA_track`, геометрический центр кузова, центр масс, центр bbox или другой anchor?
3. В каких единицах и какой мировой системе заданы `Loc_X/Y/Z` в CSV? Похоже на Unreal centimeters, но подтвердить.
4. Какая система осей/rotation order использована для camera transforms? Можно ли получить исходный Unreal transform convention или экспорт camera matrices?
5. Sensor size 23.76×13.365 mm и focal length полностью описывают intrinsics? Каковы principal point, distortion, filmback, crop/overscan?
6. Видеопотоки гарантированно синхронны и имеют общий frame 0? Есть ли PTS/time offsets?
7. Нужна 3D оценка на каждом кадре (30 Hz), с меньшей частотой или только на специальных timestamps?
8. Нужно ли выдавать 3D координаты на однокамерных сегментах? Если да, предполагается ли использование 3D модели?
9. Какой exact output format требуется: CSV columns, delimiter, units, frame/time semantics, decimal precision?
10. Как будет оцениваться качество: MAE/RMSE/threshold/percentage of frames/tracking metric? Какой допуск считается проходным?
11. На проверочном наборе будет тот же грузовик/5 камер/тот же маршрут, либо новые объекты/камеры/сцены?
12. Есть ли ограничение на GPU/CPU, Python, размер модели и время выполнения?
13. Разрешены ли pretrained open-source weights / fine-tuning / внешние библиотеки? Есть ли ограничения на доступ в интернет?
14. Можно ли использовать данные Ground Truth только на выданном тренировочном наборе; как устроен hidden test?
15. Можно ли получить исходный `object_transform_SM_AA_track.csv` без Excel date corruption в двух ячейках?

---

# 10. РАСПИСАНИЕ И СДАЧА ИЗ ПРЕЗЕНТАЦИИ

В «Зуенок А..pptx», слайд 10, указано:

- 3 ноября — выдаётся тестовый набор;
- 3–4 ноября — команды разрабатывают гипотезы и алгоритмы;
- 4 ноября, 20:00 — выдаётся проверочный набор и ссылка на Google Form;
- 4 ноября, до 24:00 — первый дедлайн: заполнение таблицы и Google Form;
- 5 ноября, до 06:00 — второй дедлайн, штраф по очкам.

Это расписание отражает выданную презентацию; команда должна следить за актуальными сообщениями организаторов и уточнить timezone/точные формы сдачи.

Критически важно до выдачи hidden/verification dataset иметь автоматический запуск без ручной подстройки по кадрам.

---

# 11. РЕКОМЕНДУЕМОЕ РАСПРЕДЕЛЕНИЕ ПЯТИ УЧАСТНИКОВ

| Участник | Зона | Ожидаемый deliverable |
|---|---|---|
| 1 — очно | data ingest, видео, pipeline, timestamps | пять источников работают, offline reader не теряет кадры, корректная синхронизация |
| 2 — очно | detector и per-camera tracking | truck detections/tracks, confidence, LOST/reacquire, overlays |
| 3 — очно | camera geometry, Unreal conversion, triangulation | validated K/R/t, dynamic cam5, 3D reconstruction |
| 4 — удалённо | 3D model / monocular PnP / target anchor research | monocular solution if model arrives; иначе documented limits and interface |
| 5 — удалённо | evaluation, ground truth, visualization, output format | metrics/CSV/plots/dashboard, GT anomaly handling |

Перед параллельной работой согласовать DTO/API:

```python
Detection2D = {
    "camera_id": "cam1",
    "frame_id": 150,
    "timestamp": 5.0,
    "bbox_xyxy": [x1, y1, x2, y2],
    "keypoints": {"anchor": [u, v]},
    "confidence": 0.94,
    "status": "TRACKING"
}

Estimate3D = {
    "frame_id": 150,
    "timestamp": 5.0,
    "position_xyz_m": [x, y, z],
    "method": "triangulation",  # monocular_pnp / predicted_only / none
    "status": "MEASURED",      # PREDICTED_ONLY / UNOBSERVABLE
    "camera_ids": ["cam2", "cam3"],
    "reprojection_error_px": 1.2,
    "confidence": 0.93,
    "uncertainty_xyz": [sx, sy, sz]
}
```

Ключевой момент: `camera_ids`, `method`, `status`, timestamps и units должны быть сохранены — это необходимо для правильной оценки.

---

# 12. ПРЕДЛАГАЕМЫЙ РЕАЛИСТИЧНЫЙ ПЛАН НА ОСТАВШЕЕСЯ ВРЕМЯ

Приоритет не в том, чтобы создать максимально много модулей. Сначала нужен доказанный end-to-end baseline.

## Спринт 1 — первые часы

1. Запустить все 23 существующих теста и smoke check.
2. Проверить пять видео и шесть CSV; создать manifest.
3. Прочитать CSV корректно, выявить corrupt GT values.
4. Реализовать camera importer.
5. Сделать projection sanity для нескольких кадров/камер.
6. Сохранить 2D overlays с Ground Truth/bbox ориентировкой.

**Выход:** доказана либо опровергнута корректность coordinate transform.

## Спринт 2 — основной baseline

1. Исправить detector on file source.
2. Запустить pretrained truck detection на всех камерах.
3. Сохранить per-camera detections.
4. Проверить coverage/false detections и bbox/anchor consistency.
5. Построить таблицу, на каких кадрах >=2 камеры одновременно видят truck.

**Выход:** валидные 2D наблюдения и visibility map.

## Спринт 3 — 3D reconstruction

1. Интегрировать dynamic cam5 pose.
2. Реализовать cross-camera association для одного tracked target.
3. Выполнить triangulation на overlap intervals.
4. Сверить с GT, когда определён правильный 3D anchor и исправлены GT anomalies.
5. Оценить coverage и ошибки по pair/subset.

**Выход:** первый 3D baseline на реальном тестовом наборе.

## Спринт 4 — single-view gaps и robust tracking

1. Проверить наличие 3D model.
2. Если получена — реализовать 2D-3D keypoint/PnP baseline.
3. Если нет — маркировать однокамерные участки как unobservable/predicted only; применить ограниченное прогнозирование и показывать uncertainty.
4. Добавить LOST/reacquire и измерить время восстановления.

**Выход:** корректная обработка всех состояний видимости, без фальшивых измерений.

## Спринт 5 — сдача

1. Сверить output schema с Google Form/template организаторов.
2. Автоматизировать batch inference на verification dataset.
3. Экспортировать нужные frame/time/XYZ/units.
4. Создать отчёт и визуализацию.
5. Запустить smoke test на чистом окружении.
6. Не менять модель/порог после hidden-test выдачи без систематической причины и без сохранения backup.

---

# 13. КАКОЙ РЕЗУЛЬТАТ ДОЛЖЕН ВЕРНУТЬ CODING AGENT

В конце работы обязательно предоставить:

1. Список изменённых/новых файлов с назначением.
2. Краткое объяснение архитектуры после изменений.
3. Точные команды установки/запуска.
4. Результат существующих 23 тестов + новые тесты.
5. Результат запуска на truck dataset.
6. Численные метрики, включая MAE/RMSE/P95/P99 и coverage, с единицами.
7. Ссылку на/путь к CSV/JSON/plots/overlays, созданным тестом.
8. Отдельный список нерешённых блокирующих вопросов.
9. Явное описание того, что не получилось реализовать и почему.
10. Подтверждение, что Ground Truth не использовался для построения predicted estimates.

Не сообщай «готово» только потому, что файлы созданы. Готовность подтверждается выполненными тестами и фактическим результатом на входных данных.

---

# 14. КОРОТКИЙ ИТОГ ДЛЯ AI

**Существующий PecStorm подходит как основа. Его не нужно переписывать.** Уже реализованы общая мультикамерная геометрия, DLT/лучевая триангуляция, reprojection refinement, эпиполярное сопоставление, Hungarian association, 3D Kalman, RTS smoothing, метрики и synthetic tests.

Для фактического задания Пеленга первоочередно не хватает:

1. корректного импорта пяти Unreal camera CSV и преобразования координат/единиц;
2. покадровой позы CameraActor5;
3. запуска detector на обычных MP4/file sources;
4. truck-specific detection + tracking + LOST/reacquire;
5. согласованной 2D keypoint/3D anchor semantics;
6. корректной обработки участков с одной камерой;
7. evaluation против Ground Truth с coverage и обработкой двух испорченных ячеек;
8. exporter по уточнённому формату организаторов.

**Главная архитектурная развилка:** если грузовик виден минимум в двух камерах — применима имеющаяся multi-view triangulation после правильной калибровки и соответствия одной и той же физической точки. Если виден только в одной камере — одна 2D точка не определяет метрическую глубину. Для полноценной 3D оценки на таких интервалах нужен предоставленный 3D truck model / 2D–3D keypoints / другой подтверждённый геометрический prior. Без этого допустим только ограниченный прогноз с честным статусом `PREDICTED_ONLY`, но не достоверная триангуляция.

**Работай по приоритетам P0 → P1 → P2, проверяй каждое изменение тестом и не заменяй исходные материалы предположениями.**

---

# 15. ИСТОЧНИКИ И ГРАНИЦЫ ДОСТОВЕРНОСТИ

- `Зуенок А..pptx`: слайды 2, 7–10 — конкурсное ТЗ, базовая цепочка, структура данных, образец результата, порядок/дедлайны.
- `Беспальцев А..pptx`: слайды 10–14 — проблемы сопровождения, архитектурные подходы, оценка tracker quality; слайды 5–9 относятся к визуальной навигации и не являются непосредственным ТЗ на грузовик.
- `project2(1).7z`: исходный датасет; пять видео и шесть CSV.
- `PecStorm-main.zip`: исходная кодовая база команды.
- `README.md` и `HANDOFF.md`: историческое описание проекта и его synthetic baseline.

**Не подтверждено документами и должно оставаться открытым вопросом:** exact target point/pivot, exact coordinate units and handedness, exact output schema, official accuracy threshold, whether 3D model will be supplied, whether every timestamp must have 3D output, whether camera videos share exact hardware timestamps, whether the inferred GT decimal repairs are correct.
