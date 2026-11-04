# Быстрый старт: как работать с проектом

Короткая инструкция «что запускать и что смотреть». Полные детали —
[docs/TRUCK_DATASET.md](TRUCK_DATASET.md), числа прогона — [docs/RESULTS.md](RESULTS.md),
открытые вопросы организаторам — [docs/OPEN_QUESTIONS.md](OPEN_QUESTIONS.md).

## 0. Окружение

* Python: запускать через лаунчер **`py`** — он уже указывает на нужный
  интерпретатор (проверено: `py -V` → 3.14.10; какой именно путь — покажет
  `py -0p`). Явный вариант — `py -3.14`. Команду `python` не использовать: на
  PATH может стоять заглушка Microsoft Store (`WindowsApps\python.exe`), она падает.
* Зависимости: `py -m pip install -r requirements.txt` (пины проверены: numpy 2.5.4,
  opencv 5.0.0, scipy 1.18.1, matplotlib 3.11.2, ultralytics 8.4.201, torch 2.15.0).
* Веса `yolov8x.pt` / `yolov8n.pt` лежат в корне и в git не входят
  (`.gitignore`); при отсутствии ultralytics скачает их сам.
* Все команды — из корня репозитория (каталог, где лежит `pecstorm.cmd`).

Проверка, что ядро живо (≈3 с):

```powershell
py -m unittest discover -s tests     # сейчас 90 тестов, OK
```

## 1. Данные

```
scene1/zadania/camera/        5 MP4 (1920×1080, 30 fps, кадры 0..1000)
scene1/zadania/xyzpowcameta/  5 camera CSV (CP1251) + GT трека
datasets/truck/                       manifest, калибровка якоря, sidecar-репары
```

Исходные файлы не изменяются: manifest только читает их и считает sha256.

## 1a. Единая команда `pecstorm`

В корне репозитория лежит `pecstorm.cmd`: одна команда вместо запоминания путей
и интерпретаторов (сам выбирает `.venv-rocm`, если он есть).

```powershell
pecstorm --help                 # список всех команд
pecstorm launch                 # веб-пульт, http://127.0.0.1:8000/
pecstorm launch --stop          # остановить пульт
pecstorm status                 # запущен ли пульт (код 0/1)

pecstorm import-data            # manifest датасета
pecstorm detect --start 0 --end 100 --step 20
pecstorm reconstruct --detections out/truck/detections.csv
pecstorm evaluate --estimates out/truck/estimates.csv
pecstorm plot-gt                # график «оценка vs эталон» (GT)
pecstorm dashboard-video --out out/truck/videos/dashboard_gt.mp4 --gt
pecstorm visualize --estimates out/truck/estimates.csv --frames 400,600
pecstorm video --detections out/truck/det_ens_v4b_v2v1.csv `
    --estimates out/truck/est_ens_v4b_v2v1.csv
pecstorm report                 # сводка + графики
pecstorm charts                 # сравнительные графики конфигураций
pecstorm labels --extract-images
pecstorm merge --inputs a.csv,b.csv --out merged.csv
pecstorm train --model yolov8s.pt --epochs 30 --device 0
pecstorm eval-model --model runs/.../best.pt --tag v5 --device 0
pecstorm anchor-views --detections out/truck/detections.csv
pecstorm anchor-azimuth --detections out/truck/detections.csv
pecstorm smoke                  # проверка ядра
pecstorm test                   # все тесты
```

Любая команда принимает свои флаги (они передаются исходному скрипту), и у
каждой есть `--help`, например `pecstorm detect --help`.

Чтобы `pecstorm` работал из любого каталога, добавьте корень репозитория в PATH:

```powershell
setx PATH "$env:PATH;<путь к репозиторию>"
# подставьте каталог, где лежит pecstorm.cmd; новое окно cmd после этого
```

## 2. Полный цикл (5 стадий + отчёт)
```powershell
# 1. Инвентаризация: manifest, проверка кадров/CSV/GT (секунды)
py apps/import_truck_dataset.py

# 2. Детекция YOLO по всем кадрам. ДОЛГО: ~53 мин CPU для 1001 кадра × 5 камер
py apps/detect_truck.py --start 0 --end 1000 --step 1 `
    --model yolov8x.pt --conf 0.25 --imgsz 960 `
    --out out/truck/detections.csv --stats out/truck/detect_stats.json

# 2a. То же на GPU (AMD/Windows через DirectML): ≈3.5 мин вместо 53
#     нужны yolov8x.onnx (YOLO('yolov8x.pt').export(format='onnx', imgsz=960))
#     и pip install onnxruntime-directml
py apps/detect_truck.py --start 0 --end 1000 --step 1 `
    --model yolov8x.onnx --provider dml --conf 0.25 --imgsz 960 `
    --out out/truck/detections_dml.csv --stats out/truck/detect_stats_dml.json

# 2b. Калибровка якоря — под тот CSV, на котором считаете (после смены бэкенда
#     боксы чуть другие, якоря надо пересчитать)
py tools/anchor_fit_views.py --detections out/truck/detections_dml.csv `
    --max-reprojection-px 50

# 3. Калибровка якоря по наборам видов (секунды; файл уже в репозитории)
py tools/anchor_fit_views.py --detections out/truck/detections.csv `
    --max-reprojection-px 50

# 4. Реконструкция 3D по готовым детекциям (<1 с)
py apps/reconstruct_truck.py --detections out/truck/detections.csv `
    --max-reprojection-px 50 `
    --out-csv out/truck/estimates.csv --out-json out/truck/estimates.json

# 5. Оценка против GT (ошибка и покрытие — раздельно)
py apps/evaluate_truck.py --estimates out/truck/estimates.csv `
    --out-json out/truck/evaluation.json --out-csv out/truck/eval_per_frame.csv

# 6. Отчёт: сводка JSON + графики, включая «Итоговый результат» X/Y/Z
py tools/make_report.py
```

Пока нет полной детекции, для быстрой проверки ставьте `--step 20` (в 20 раз
быстрее) или `--start 0 --end 200`; тогда и дальше передавайте тот же CSV.

## 3. Куда смотреть результат

| файл | что показывает |
|---|---|
| `out/truck/estimates.csv` / `.json` | оценки 3D: `frame`, `x,y,z`, `vx,vy,vz`, `status`, `method`, `views`, `confidence` (DTO §11) |
| `out/truck/evaluation.json` | MAE/RMSE/median/P95/P99/max + coverage по статусам |
| `out/truck/eval_per_frame.csv` | построчно: оценка, GT, ошибка, статус |
| `out/truck/report/summary.json` | сводка всех стадий одним файлом |
| `out/truck/report/components_vs_time.png` | «Итоговый результат»: X/Y/Z, эталон vs оценка |
| `out/truck/report/components_error_vs_time.png` | то же + расхождение по осям |
| `out/truck/report/trajectory_xy.png` | траектория в плане |
| `out/truck/viz/*.png` | проекция оценки и GT на кадрах камер |
| `out/experiments/*.json` | видимость, проекция bbox↔pivot, свип порогов, робастность, бенчмарк |

Только «итоговый» график, без полного отчёта:

```powershell
py tools/plot_components.py --estimates out/truck/estimates.csv
# добавить --no-predicted, чтобы осталось ровно 6 линий, как в презентации
```

## 4. Как читать числа (важно)

* **MAE считается только по `MEASURED`** (≥2 вида). Это не «точность по всем
  кадрам»: покрытие сейчас 61.3 %, ещё 36.2 % кадров без оценки.
* Эталонный MAE 4.28 м — **in-sample** по калибровке якоря. Честная оценка
  переноса — 4.5 м (проверка чётные/нечётные кадры), см. `docs/RESULTS.md` §5.
* Без файла `datasets/truck/anchor_offset_views.json` MAE падает до 11.9 м —
  калибровку нельзя терять (файл должен быть в git).
* GT — pivot актора, детектор — центр bbox: расхождение 57–229 px по камерам.
  Поэтому «идеально ровной» линии на графике не будет без 3D-модели объекта.

## 4a. Дообучение детектора на этой сцене (GPU, ROCm)
Псевдо-разметка из GT + 3D-модели (учитель), затем обучение в `.venv-rocm`:

```powershell
# 1. разметка (боксы из модели + GT), картинки выгружаются один раз
py tools/make_pseudo_labels.py --out datasets/truck/pseudo --extract-images

# 2. (опционально) hard-negative кадры: ложные срабатывания там, где цели нет
py tools/make_hard_negatives.py --detections out/truck/detections_ft.csv

# 3. обучение (GPU; для CPU уберите --device 0)
.\.venv-rocm\Scripts\python.exe tools\train_truck_detector.py `
    --model yolov8s.pt --epochs 30 --imgsz 960 --batch 8 --device 0 `
    --patience 8 --name scene_v1

# 4. полный цикл оценки: детекция -> якоря -> 3D -> MAE/покрытие/recall-прокси
.\.venv-rocm\Scripts\python.exe tools/run_finetuned_eval.py `
    --model runs/detect/out/train/scene_v1/weights/best.pt --tag v1 --device 0
```

Данные для обучения пишутся в `datasets/truck/pseudo/` (в git не входят, это
тысячи JPEG). Метрики последнего прогона — `docs/RESULTS.md` §8d.

## 4b. Видео с наложением трекинга — как на пульте
```powershell
py tools/render_tracking_video.py `
    --detections out/truck/det_ens_v4b_v2v1.csv `
    --estimates out/truck/est_ens_v4b_v2v1.csv `
    --out out/truck/videos
    # --step 2 --fps 15 --cameras cam3,cam5 --scale 0.5  — короче/легче
    # --gt — добавить жёлтый GT-пивот; --no-map — без map_plan.mp4
```

Пишется по MP4 на камеру **плюс** `map_plan.mp4` — панель «план X–Y (мир)».
Оформление повторяет веб-пульт: голубые боксы детекций, толстый сглаженный путь
(зелёный — измерения, оранжевый — прогноз/достройка), белый пунктир — прогноз
вперёд, серый пунктир — пропадание данных, 3D-маркер со статусом, HUD и легенда.
Стиль задаётся флагами `--trail 160 --smooth 5 --forecast-s 3 --bridge-s 3`
(значения по умолчанию совпадают с пультом).

Запись «как веб-пульт» — одним видео: 5 камер + план X–Y + таблица метрик:

```powershell
py tools/render_dashboard_video.py `
    --out out/truck/videos/dashboard_gt.mp4 --gt
```

В таблице — **фактические** покрытие и MAE по накопленным кадрам (полный клип:
82.3 % / 1.98 м). Целевых отметок в видео нет, только факт. Фрагмент 0..213
честно даёт 93.0 % покрытия (199/214 кадров), но MAE на нём 2.77 м — это
фрагмент, а не весь клип.

График сравнения с эталоном:

```powershell
pecstorm plot-gt
# -> results/truck/report_final/comparison_vs_gt.png (+ .json с числами)
```

Слева — траектория X–Y: чёрная линия GT и точки оценки, цвет — ошибка; справа —
ошибка по кадрам, MAE и зона holdout. Числа берутся из
`out/truck/eval_final/evalpf.csv` и `eval.json`, GT — из датасета.

## 4c. Веб-пульт и трекинг в реальном времени

```powershell
.\.venv-rocm\Scripts\python.exe apps\dashboard.py `
    --model runs\detect\out\train\scene_v4b_ft\weights\best.pt `
    --anchor-views datasets\truck\anchor_offset_views_v4b.json `
    --anchor-azimuth datasets\truck\anchor_azimuth_v4b.json `
    --device 0 --imgsz 960 --preview-scale 0.5 --port 8000
# открыть http://127.0.0.1:8000/
# по умолчанию (pecstorm launch) уже используются v4b и её якоря
```

Что умеет:

* читает 5 MP4 «Пеленги» в темпе, близком к реальному, и гоняет полный конвейер
  (детекция → слияние → трекинг) на каждом кадре;
* отдаёт 5 MJPEG-потоков камер + панель плана X–Y: боксы детекций, зелёный
  маркер 3D-оценки, HUD;
* кнопки: пауза/продолжить, `conf ±`, **`Эталон`** — включает жёлтую линию GT
  на плане X–Y и жёлтый крест GT на камерах (сравнение с идеалом на лету);
* JSON-статистика `/api/stats` (кадр, fps, доля измеренных кадров, последняя
  3D-позиция, состояние эталона).

Замеренная производительность (RX 7900 XT, imgsz 960, 5 камер,
`--preview-scale 0.5`; замеры 04.11.2026):

| режим | fps |
|---|---|
| штатный: **v4b** (yolov8m), путь, карта, сглаживание, прогноз | **9.2–9.5** |
| быстрый резерв: v1 (yolov8s), те же наложения | 10.5–12.0 |

v4b выбрана для пульта, потому что её покрытие заметно выше: в демо-прогоне
`measured_share` был 87–95 % (у v1 на тех же кадрах — около 60 %). Офлайн-метрика
v4b по всему клипу — 81.7 % покрытия (см. `docs/RESULTS.md` §8d); ансамбль
v1+v2+v4b (82.3 %) в реальном времени не тянет.

Траектория и прогноз:

* зелёная толстая линия — измеренные 3D-точки трека (`MEASURED`);
* оранжевая — прогноз: Калман на кадрах без измерения (`PREDICTED_ONLY`) и
  достройка после пропадания (`EXTRAPOLATED`);
* белый пунктир — прогноз вперёд: продолжение по средней скорости последнего
  непрерывного участка измерений на `--forecast-s` секунд (по умолчанию 3 с);
* серый пунктир — разрыв данных: соединяет концы пропадания, если разрыв не
  дальше 120 м. При скачке трека больше 10 м линия рвётся без соединителя;
* жёлтая линия/крест — эталон GT (кнопка «Эталон»);
* `--smooth N` — сглаживание пути фильтром Савицкого–Голея по окну ±N точек
  (по умолчанию 5; 0 — выключить). Края продолжаются линейно, поэтому
  равномерное движение не «отстаёт»;
* `--trail N` — сколько точек рисовать на видео (по умолчанию 160);
* `--map-trail N` — сколько точек рисовать на плане X–Y (по умолчанию 0 —
  вся история);
* `--max-missing N` — сколько кадров Калман продолжает трек без измерения
  (по умолчанию 5, чтобы маркер не уезжал);
* `--bridge-s S` — сколько секунд достраивать путь после последнего измерения
  (по умолчанию 2; 0 — выключить).

Отдельная панель **«план X–Y (мир)»** — первая плитка в интерфейсе: путь объекта
в мировых координатах с сеткой, метками осей, точкой старта, текущим положением
и прогнозом вперёд. Окно само подстраивается под путь (границы по 2–98 %
перцентилям, расширяется сразу, сжимается плавно), шаг сетки — 10/25/50 м.
Панель строится сервером как ещё один MJPEG-поток (`/stream/map`), поэтому
работает без JavaScript-графики.

Таблица прогона (после запуска пульта):

```powershell
pecstorm dashboard-report --seconds 180
# -> results/truck/report_final/dashboard_run.png (+ .csv, .json, .md)
```

Скрипт ждёт завершения полного прохода и пишет фактические fps, покрытие и MAE
против GT; ничего не «улучшает». Если пульт уже прошёл круг, он дождётся
следующего.

CSV координат оценки vs эталона по пробегу (из данных `dashboard_gt.mp4`):

```powershell
pecstorm coords-table
# -> results/truck/report_final/dashboard_coords_sampled.csv  (каждые 10 м)
#    results/truck/report_final/dashboard_coords.csv          (все 1001 кадр)
```

В CSV колонки: пробег, кадр, время, статус, X/Y/Z оценки, X/Y/Z эталона,
dX/dY/dZ и ошибка. Кодировка UTF-8 с BOM — Excel открывает без кракозябр.

То есть «реальное время» здесь — это ~9–9.5 кадров/с на 5 камер (v4b). Честно:
**30 fps не держим**. Чтобы поднять: уменьшить `imgsz`/число камер, вынести
кодирование в отдельный поток или батчить детекцию; ансамбль из трёх моделей
в реальном времени не тянет — он только для офлайн-прогонов.

## 5. Если данных станет больше / другая сцена

1. Положить новые MP4 и camera CSV в те же подпапки `scene1/zadania/`.
2. `py apps/import_truck_dataset.py` — проверить отчёт о расхождениях.
3. `python apps/detect_truck.py ...` (полный прогон).
4. `python tools/anchor_fit_views.py ...` — **обязательно** перекалибровать
   сдвиги (они зависят от сцены/камер).
5. Дальше — стадии 4–6 как выше.

Пороги слияния и модель детектора меняются в `config/truck.json`
(`fusion.max_reprojection_px`, `detector.model`, `detector.imgsz`).

## 6. Грабли (проверено)

* PowerShell показывает кириллицу как mojibake; для корректного вывода —
  `$env:PYTHONIOENCODING="utf-8"` перед запуском.
* `unittest` пишет в stderr, поэтому PowerShell возвращает код 1 даже при
  «OK» — смотреть на строку `Ran N tests ... OK`.
* Полная детекция ~53 мин CPU; параллельные тяжёлые прогоны её замедляют.
* `--video` у `apps/visualize_truck.py` пишет montage PNG, а не mp4.
* JSON-файлы читаются с BOM (`utf-8-sig`) — файлы из PowerShell не ломают загрузку.
