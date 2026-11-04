# HANDOFF — состояние работ по проекту multiview3d

> **ОБНОВЛЕНИЕ 03.11.2026 (сессия 4).** Полный прогон по всем 1001 кадру выполнен,
> числа зафиксированы в [docs/RESULTS.md](docs/RESULTS.md). Коротко: детекция
> 93.7 % кадров (1660 детекций, 5005 кадро-камер, 3200 с на CPU); покрытие
> 61.3 % `MEASURED`; MAE 4.28 м (медиана 2.91 м, P95 11.18 м) с калибровкой
> якоря **по наборам видов** (`datasets/truck/anchor_offset_views.json`).
> Честная проверка переноса (чётные/нечётные кадры) — 4.5 м для двух основных
> пар видов. Общий 3D-сдвиг и 2D-сдвиг по камерам реализованы, но на holdout
> не переносятся (негативные результаты в §5 RESULTS). Тестов — 69, все
> зелёные. Статус см. §0.5.

> **ОБНОВЛЕНИЕ 03.11.2026 (сессия 3).** Truck-workflow собран целиком: ML-детектор
> (YOLO), позы камер по кадру (cam5 вращается), триангуляция, статусы DTO §11,
> экспорт оценок, оценка против GT с раздельными ошибкой и покрытием, 63 теста
> `unittest` (все зелёные), эксперименты, документация.
>
> Ключевые документы: [docs/TRUCK_DATASET.md](docs/TRUCK_DATASET.md) (формат,
> конверсия, ограничения), [docs/OPEN_QUESTIONS.md](docs/OPEN_QUESTIONS.md)
> (15 вопросов организаторам, ответов нет), [docs/RESULTS.md](docs/RESULTS.md)
> (измеренные результаты).
>
> Что важно: конвенция координат остаётся **неподтверждённой**, 2D-якорь
> детектора (центр bbox) не равен GT-pivot, поэтому есть калибруемый на
> train-кадрах постоянный сдвиг `anchor_offset_m`. Численная точность на
> реальных данных ограничена этим и неподтверждёнными intrinsics; «порог
> прохождения» организаторами не задан (вопрос №10).
>
> Текст ниже (§0 — сессия 2) сохранён как история; часть пунктов «доделать» из
> §0.3 уже закрыта в сессии 3 — фактический статус см. в §0.4.

> **ОБНОВЛЕНИЕ 03.11.2026 (сессия 2) — две линии работ.** Актуальный статус
> датасета «Пеленга» (UE-экспорт) и список оставшихся работ — в **§0** ниже.
>
> * **Синтетическое ядро завершено** (план P0–P2): `pipeline.py`, `metrics.py`,
>   `visualize.py`, `__init__.py`, `apps/` (4 CLI), `config/demo.json`,
>   `requirements.txt`, `tests/` (23 теста `unittest`); `fusion.py` переписан;
>   дефекты §4 исправлены. Критерий приёмки P0: 3 камеры / 2 объекта / 25 fps →
>   2 трека, средняя ошибка 3D **10.14 мм** ≤ 30 мм (oracle, seed=7); демо на
>   blob-детекторе: среднее 7.48 мм, полнота 1.0, ID-switch 0 (`out/metrics.json`).
> * **Датасет «Пеленга» (бриф START_HERE_MASTER_BRIEF_RU.md): ingest готов;
>   конвенция координат НЕ подтверждена; детектора грузовика НЕТ; тесты §7.1
>   брифа не написаны.** Подробности — в §0.
>
> Текст после §0 — исторический разбор исходного состояния, оставлен как справка.

## 0. Актуальный статус: датасет «Пеленга» (обновление 03.11.2026, сессия 2)

Все утверждения ниже проверены запусками команд в этот день; источники указаны.

**Источники задания:**

* исходное ТЗ (оригинал, получен в Telegram; локальная копия в репозиторий не входит):
  «Техническое задание: Система мультикамерного 3D-трекинга движущегося объекта»
  (15 932 байт, sha256 `fbaa05f695f5fa2027590bbb6e509cc3ec5d4bd5fc1086b02a55010e9b6d53ad`,
  mtime 03.11.2026 16:54 по `ls -la`);
* **полный текст ТЗ в репозитории**: `docs/TZ_hackaton_10.2026.md` — скопирован
  03.11.2026 скриптом `tools/extract_docx.py` из `word/document.xml`; полнота
  проверена побуквенно (1596 символов текста оригинала == текст копии с
  точностью до пробелов и маркеров списков); изображений в оригинале 0;
* детальный бриф/тест-план этого репозитория: `START_HERE_MASTER_BRIEF_RU.md`;
* материалы организаторов: `scene1/TZ/Беспальцев А..pptx`,
  `scene1/TZ/Зуенок А..pptx`, данные — `scene1/zadania/`.

### 0.1 Сделано и проверено

1. **Ingest UE-экспорта** — `multiview3d/unreal_camera.py`: чтение CP1251/`;`
   (камеры и GT), парсинг focal length и размера сенсора, детект битых ячеек без
   silent-coercion, sidecar-репары с флагом `verified_by_organizer`, pinhole
   intrinsics, экстринсика OpenCV (эмпирическая конвенция, docstring файла).
   `multiview3d/camera_provider.py` — `CameraStateProvider`: позы по номеру кадра
   (cam1–4 статичны, cam5 per-frame, без интерполяции углов).
2. **Аудит датасета** — `tools/audit_truck_dataset.py` → `datasets/truck/dataset_manifest.json`:
   * 5 MP4: 1920×1080, 30 fps, `frames_property == frames_decodable == 1001`
     (кадры 0..1000);
   * 5 CSV камер: 1002 строки, кадры 0..1001 (строка 1001 — лишняя относительно видео);
   * cam1–4 статичны; cam5: yaw ∈ [−179.8084°, +179.7232°], 1 wrap-скачок,
     суммарный развёрнутый yaw −92°, максимальный шаг поворота 0.5227°/кадр;
   * GT CSV: 1000 строк; обнаружены 2 повреждённые ячейки: frame 402 `Loc_X`
     «авг.13», frame 691 `Loc_Y` «июл.25».
3. **Sidecar-репары** — `datasets/truck/ground_truth_repairs.json`: гипотезы
   8.13 / 7.25 (Excel date coercion, формат mm.dd); `verified_by_organizer: false`,
   `accept_unverified: false` — исходные файлы не изменены.
4. **Projection sanity** — `apps/projection_sanity.py`: проекция GT-пивота во все
   5 камер, кадры 0..900 шаг 150 → `out/truck_sanity/overlay_cam{1..5}_frame*.png`
   и карта видимости `out/truck_sanity/visibility.json` (на cadence 30 кадров
   обычно видны cam1, cam4, cam5; cam2/cam3 подключаются на части интервалов).
5. **Инструенты подбора конвенции** — `apps/transform_probe.py` (перебор
   16 sign-комбинаций), `tools/convention_check.py`, `tools/pose_fit.py`,
   `tools/anchor_fit.py`.
6. **Тесты синтетического ядра перепроверены сегодня**:
   `python -m unittest discover -s tests -q` → `Ran 23 tests — OK` (1.651 с).
   `pytest` в интерпретаторе **не установлен** (проверка `find_spec` — False);
   тесты запускаются через `unittest`.

### 0.2 Не подтверждено — поз-фит даёт большие невязки

Фактический вывод `tools/pose_fit.py` (запуск 03.11.2026):

| cam | d_yaw | d_pitch | focal_scale s | mean | max (px) |
|---|---|---|---|---|---|
| 1 | +2.95 | −1.12 | 0.922 | 6.8 | 13.4 |
| 2 | +0.08 | −0.78 | 1.084 | 25.7 | 72.5 |
| 3 | −2.43 | −0.82 | 0.953 | 22.2 | 39.7 |
| 4 | +1.83 | −0.53 | 1.571 | 81.5 | 176.3 |
| 5 | −0.31 | −3.36 | 2.442 | н/д | н/д (1 якорь — фит вырожден: 3 параметра, 2 остатка) |

`tools/anchor_fit.py` (якорь = точка касания колёс): max невязка 214.1 px (so=+1)
и 227.9 px (so=−1) → согласованный 3D-сдвиг якоря **не подтверждён**.

Вывод: эмпирическая конвенция (знак pitch зеркален относительно CSV) примерно
согласуется с cam1 (6.8 px), но **не подтверждена для датасета в целом**;
fit-ные focal scale 0.92..1.57 указывают на проблему в допущении intrinsics
(сенсорный crop/overscan или principal point). Утверждать, что реальная
калибровка зафиксирована, нельзя. Якоря фитов — ручные оценки по кадрам,
их точность не верифицирована.

### 0.3 Сверка с master-брифом: статус и список доделок

**Все шаги брифа `START_HERE_MASTER_BRIEF_RU.md` НЕ выполнены.** Готовы базовая
линия, аудит и ingest; основная часть задания (детекция → 3D → оценка → сдача)
не сделана.

**Выполнено (подтверждено запусками 03.11.2026):**

- Шаг A baseline (§0.3 брифа): `python -m unittest discover -s tests -q` → 23 OK;
  `tools/smoke_check.py` → rc=0 (2 трека, среднее 10.136 мм, p95 27.31 мм,
  ID-switch 0, 356 fps).
- Шаг B аудит данных (§6 P0.1): `datasets/truck/dataset_manifest.json`.
  Замечание: параметры видео получены через OpenCV; `ffprobe` из брифа не запускался.
- Шаг C ingest: `unreal_camera.py` + `camera_provider.py` — код есть (тестов нет,
  конвенция не подтверждена — §0.2).
- Спринт 1 (§12): projection sanity оверлеи по 5 камерам.

**Доделать — deliverable по тест-плану (§7):**

1. **§7.1 — 9 обязательных тестов данных/камеры** (`test_camera_csv_parse`,
   `test_gt_csv_invalid_cells`, `test_gt_override_audit`, `test_camera_units`,
   `test_unreal_to_opencv_axes`, `test_static_camera_pose`,
   `test_camera5_yaw_unwrap`, `test_video_frame_alignment`,
   `test_frame_count_mismatch`). Фактически в `tests/` тестов на
   `unreal_camera`/`camera_provider` **нет** (grep по tests/ — 0 совпадений).
   Задача: компактный файл `tests/test_unreal_ingest.py`.
2. **§7.2 — projection sanity как автоматический тест** (сейчас только ручная
   диагностика `apps/projection_sanity.py`, включая cam5 в повороте).
3. **Зафиксировать конвенцию координат**: проверить допущения focal/sensor
   (UE sensor crop/overscan, principal point), перепроверить якоря; запросить
   подтверждение у организаторов. Задокументировать в `docs/TRUCK_DATASET.md` и
   `docs/OPEN_QUESTIONS.md` — на них ссылается docstring `unreal_camera.py`,
   этих файлов **нет** (в `docs/` пока только `TZ_hackaton_10.2026.md`).
4. **Границы кадров**: видео 1001 кадр (0..1000), CSV имеет строку frame 1001 —
   обработка общей области 0..1000 без off-by-one (§7.1 п.9).
5. **§7.3 — детектор грузовика**: в репозитории нет ни одного `.onnx`
   (`find` — пусто); `ultralytics`/`torch` не установлены (проверка `find_spec`),
   `onnxruntime` установлен (1.31.1). Нужен источник модели (веб-поиск
   пользователем запрещён — запросить путь/источник у пользователя); затем
   ручная stratified-разметка выборки, precision/recall/IoU per camera,
   overlays FP/FN.
6. **§7.4 — reconstruction tests**: парные камеры, шум 2D 0.25..5 px,
   perturbation калибровки, reject ill-conditioned.
7. **§7.5 — tracking tests**: окклюзии, пропадания, ID-switch/reacquisition.
8. **§7.6 — метрики на реальных данных с coverage** (MAE/RMSE/P95/P99, доля
   кадров с ≥2 views / monocular / predicted-only); `out/metrics.json` сейчас —
   только синтетическое демо.
9. **§7.7 — интеграционный acceptance**: конвейер не подключён к
   `CameraStateProvider` и mp4-источникам (`config/demo.json` — синтетика);
   нужен прогон 5 видео → детекция → reconstruction → экспорт с GT-сравнением
   после явного разрешения репаров.
10. **Подтверждение репаров GT** у организаторов (`verified_by_organizer=false`).

**Доделать — дефекты кода из брифа §4 (подтверждено чтением кода 03.11.2026):**

- **P0: детектор не запускается на file/stream** — `pipeline.py:155-157` создаёт
  `OpenCvFrameSource` без детектора; нет adapter-декоратора `DetectingFrameSource`
  (§6 P0.3).
- **P0: конвейер статичен по камерам** — `CameraStateProvider` не интегрирован в
  pipeline/fusion (grep: используется только в `apps/projection_sanity.py`)
  (§6 P0.5).
- **P0: `read_aligned` требует кадры всех камер** — `capture.py:197-203`
  возвращает `None` при нехватке любой камеры; partial availability не поддержана
  (§6 P1.1).
- **P0/P1: `LatestFrameBuffer.get()` возвращает старейший** — FIFO `Queue.get()`
  (`capture.py:73-75`), latest-policy нет.
- **P1: тайминги** — время меряется только вокруг `tracker.step()`
  (`pipeline.py:168-170`); end-to-end/per-stage timings нет.
- **P1: 2D anchor/keypoint семантика** не определена (§6 P0.4, §9 п.2 — вопрос
  организаторам).
- **P1/P2: статусы `MEASURED/PREDICTED_ONLY/UNOBSERVABLE`, method, uncertainty,
  camera_ids** — полей в `types.py` нет (grep `status|method` — 0 совпадений);
  DTO из §11 не реализован.
- **P1: синхронность видео** — timestamp = index/fps, PTS не проверялся (§9 п.6).

**Доделать — отсутствующие deliverable (§0.3 шаг H, §5.1, §12–§13):**

- Приложения truck-workflow (§5.1): `apps/import_truck_dataset.py`,
  `detect_truck.py`, `reconstruct_truck.py`, `evaluate_truck.py`,
  `visualize_truck.py` — нет ни одного.
- Каталог `experiments/` (visibility_analysis, detector_benchmark, geometry_sweep,
  robustness_benchmark) — отсутствует.
- 3D baseline на GT (шаг E, спринт 3) — триангуляция на truck-данных не выполнялась.
- Обработка однокамерных участков (шаг F): статусы + monocular-интерфейс;
  3D-модель грузовика не предоставлена (§2.6, §9 п.1/п.8).
- Экспорт координат в формате организаторов (шаг H, §9 п.9) — настраиваемого
  exporter нет.
- `docs/OPEN_QUESTIONS.md` — 15 вопросов из §9 не зафиксированы;
  `docs/TRUCK_DATASET.md`, `docs/RESULTS.md` отсутствуют.
- Итоговый отчёт по §13 (список файлов, команды, truck-метрики, подтверждение
  что GT не использовался как вход) — не может быть составлен до выполнения
  пунктов 5, 8, 9 списка выше.
- README: раздел запуска на truck dataset (§0.1 п.8) отсутствует — README описывает
  только синтетическое ядро.

**Расписание сдачи (§10 брифа, из презентации «Зуенок А..pptx», слайд 10):**
проверочный набор — 4 ноября 20:00, дедлайн 4 ноября 24:00 (штраф до 5 ноября
06:00). Сегодня 3 ноября — до выдачи hidden-набора нужно довести до конца пункты
P0 (детектор на file sources, интеграция provider, тесты §7.1, базовый
3D-прогон), иначе автоматический прогон на проверочном датасете не будет готов.

### 0.4 Сессия 3: что сделано (проверено запусками)

1. **ML-детектор на реальных видео.**
   `apps/detect_truck.py` + `multiview3d/truck_pipeline.py: detect_frames`:
   последовательное чтение MP4 (`TruckVideoSource`), таймкод `frame/fps`,
   CSV 2D-детекций, статистика по камерам. Используется `YoloDetector`
   (`yolov8x.pt`, `imgsz=960`, `conf=0.25`, классы `truck/car/bus/train`).
   Факт быстрого прогона (кадры 0..200 шаг 20): cam1 — 10/11 кадров с
   детекцией, cam2 — 6/11, cam3–cam5 — 0/11 (на этом интервале грузовик им
   почти не виден); ~3.0 с на кадр всех пяти камер на CPU.
2. **Реконструкция по кадрам.** `apps/reconstruct_truck.py` +
   `build_pipeline/run_reconstruction`: позы берутся из provider **на каждый
   кадр** (cam5 вращается), слияние `association_mode="single_target"`,
   порог `max_reprojection_px=30` (bbox-якоря дают десятки px невязки),
   калибруемый `anchor_offset_m`. Смоук-прогон (кадры 0..200 шаг 20):
   5 MEASURED-точек по видам cam1|cam2.
3. **Оценка отдельно от покрытия.** `apps/evaluate_truck.py` +
   `metrics.evaluate_truck`: MAE/RMSE/median/P95/P99/max по MEASURED,
   отдельная статистика по PREDICTED_ONLY, coverage по статусам, подмножества
   по наборам видов. Смоук без сдвига якоря (кадры 80..180): MAE **12.5 м**
   при покрытии 5 % — то есть систематический сдвиг якоря, а не «шум».
4. **Калибровка якоря.** `tools/anchor_fit_yolo.py`: фит постоянного 3D-сдвига
   `project(cam_f, pivot_gt + delta) ≈ bbox_center` на train-кадрах,
   проверка на отложенных. Может работать по готовому CSV детекций
   (`--detections`). Артефакт — `datasets/truck/anchor_offset.json`.
5. **Приложения и эксперименты.** Пять truck-apps (`import/detect/reconstruct/
   evaluate/visualize`), `config/truck.json`, `experiments/`:
   `visibility_analysis.py`, `projection_sanity.py`, `detector_benchmark.py`,
   `geometry_sweep.py`, `robustness_benchmark.py`.
6. **Тесты.** `tests/test_unreal_ingest.py` (все 9 обязательных пунктов §7.1) и
   `tests/test_truck_pipeline.py` (проекция в видео, шум 2D, пропадания видов,
   `DetectingFrameSource`, динамическая cam5, coverage-метрики, конфиг).
   `python -m unittest discover -s tests` → **63 теста, OK**.
7. **Документация.** `docs/TRUCK_DATASET.md` (формат/конверсия/ограничения),
   `docs/OPEN_QUESTIONS.md` (15 вопросов §9, ответов нет), `docs/RESULTS.md`
   (измеренные числа), раздел README про truck-workflow,
   `requirements.txt` (ultralytics/torch), `.gitignore` (веса моделей).
8. **Видимость (факт).** `experiments/visibility_analysis.py --step 5`:
   ≥2 камеры видят GT-пивот на 195 из 201 кадров (97 %); по камерам cam1 15.9 %,
   cam2 76.6 %, cam3 55.2 %, cam4 77.1 %, cam5 100 %.
9. **Робастность (синтетика, факт).** `experiments/robustness_benchmark.py`:
   шум 2D 5 px → средняя 3D-ошибка 0.026 м; focal +2 % → 0.020 м;
   principal point +8 px → 0.039 м; поворот камеры 2° → 0.015 м; сдвиг метки
   времени 50 мс при 2.5 м/с → 0.12 м.

**Осталось (вне кода):** ответы организаторов (§9, особенно пункты 2, 5, 6, 9,
10); численный target точности не задан. Полный прогон детекции по 1001 кадру
(≈50 мин CPU) выполняется отдельно, его числа — в `docs/RESULTS.md`.

### 0.5 Сессия 4: полный прогон и результаты

1. **Полная детекция.** `apps/detect_truck.py --start 0 --end 1000 --step 1`
   (`yolov8x.pt`, `imgsz=960`, `conf=0.25`, CPU): 5005 кадро-камер за 3200.2 с
   (≈0.64 с на камеру-кадр), 1660 детекций, кадров с детекцией 938 (93.7 %).
   По камерам (кадров с детекцией): cam1 189 (18.9 %), cam2 231 (23.1 %),
   cam3 261 (26.1 %), cam4 301 (30.1 %), cam5 602 (60.1 %).
2. **Структура данных (факт).** Детекции cam1/cam2 есть только в 0..500,
   cam4 — почти только в 600..1000 (6 против 300). Поэтому train/holdout по
   времени разделяет и наборы камер: `cam1|cam2` существует только в первой
   половине, holdout — только `cam3|cam5` и `cam4|cam5`.
3. **Почему ошибка в метрах.** `experiments/projection_sanity.py`: расстояние
   «центр bbox ↔ проекция GT-пивота» — медиана 140.5 px (среднее 155.8 px,
   n=160), по камерам от 57.3 (cam2) до 229.3 (cam5) px. Это не шум детектора.
4. **Три модели якоря, сравнение на одних и тех же порогах** (`max_reproj=50`):
   без сдвига 11.89 м; общий 3D-сдвиг (фит train) 7.40 м (holdout 8.74);
   2D-сдвиг по камерам (фит train) 10.00 м (holdout **хуже**: 14.17);
   3D-сдвиг на набор видов 4.28 м. Основной вариант —
   `FusionConfig.anchor_offset_views` + `tools/anchor_fit_views.py`.
5. **Честность калибровки.** Сдвиги по видам фитируются по всем кадрам, поэтому
   4.28 м — in-sample. Проверка `parity` (чётные/нечётные кадры даёт 4.51 м на
   `cam3|cam5` и `cam4|cam5`, `time` (0..500 → 600..1000) — 3.65 и 2.96 м.
   Трёхвидные наборы (`cam2|cam3|cam5` n=16, `cam3|cam4|cam5` n=13,
   `cam2|cam3` n=4) не калиброваны: у них ошибка остаётся 9.6–11.9 м.
6. **Исправлены знаки калибровок.** `tools/anchor_fit_yolo.py` пишет
   `offset_m = -delta` (delta — сдвиг pivot→bbox), `anchor_px_offset`
   вычитается из центра детекции; `anchor_offset_views` прибавляется к оценке.
   До этого сдвиг применялся с обратным знаком и ухудшал MAE — числа с
   исправленным знаком в §5 RESULTS.
7. **Порог `max_reprojection_px`.** По `experiments/geometry_sweep.py` (кадры
   0..400, шаг 5) 30 → 50 px увеличивает покрытие 0.465 → 0.521 при MAE
   6.74 → 6.27 м; на полном прогоне покрытие 55.7 % → 61.3 % при MAE 4.31 →
   4.28 м. В `config/truck.json` и `apps/reconstruct_truck.py` — 50 px.
8. **Тесты.** Добавлены проверки 2D-сдвига по камерам и сдвига по наборам
   видов (канонический ключ, порядок камер, чужой набор не применяется).
   `python -m unittest discover -s tests` → **69 тестов, OK**.
9. **Документация.** Создан `docs/RESULTS.md` (детекция, видимость, якоря,
   покрытие, эксперименты, ограничения, воспроизведение); обновлены README
   (числа + команды + `detect.py`) и §6/§6a `docs/TRUCK_DATASET.md`.
10. **График «Итоговый результат».** `multiview3d.visualize.plot_components_vs_time`
    + `tools/plot_components.py`: `X_t`/`Y_t`/`Z_t` (эталон, сплошные) против
    `X_det`/`Y_det`/`Z_det` (оценка, штриховые), плюс версия с панелью
    «оценка − эталон» по осям. Строится автоматически в `tools/make_report.py`:
    `out/truck/report/components_vs_time.png` и `components_error_vs_time.png`,
    числа — в `components_vs_time.json` (MAE X=2.969, Y=2.173, Z=0.907 м;
    max |Δ| 68.02/21.36/8.84 м на n=612). Тестов — 70.

**Активные ограничения:** 4.28 м — in-sample, честная оценка ≈4.5 м;
P99 12.6 м, max 71.8 м (отдельные кадры на большой дальности); 3-видные
наборы без калибровки (7.4 м); одновидовых измерений 0 % (3D без 3D-модели
объекта не восстанавливается); ответы организаторов (вопросы 2, 5, 6, 9, 10)
по-прежнему не получены.

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
* Python запускать так: **`py`** (лаунчер, `py -V` → 3.14.10, `py -0p` →
  `...\Python314\python.exe`) или `py -3.14`. Полный путь нужен только если
  лаунчера нет. Команда `python` на PATH — заглушка Microsoft Store
  (`...\WindowsApps\python.exe`), она падает.
* Все команды из документации (README, docs/*.md) используют `py`.
* В Code Mode (`execute`) вызов `tools.shell(...)` недоступен (`Unknown tool
  'shell'`); `shell` работает только как прямой инструмент.
* Git: изменения не коммитились. `git log` — единственный коммит `2bedc37 Initial
  commit`; `multiview3d/`, `tools/`, `HANDOFF.md` — untracked, `README.md` — изменён.
  Временные `_probe*.py` удалены.

## 7. Как воспроизвести проверки

```powershell
py tools\smoke_check.py
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
