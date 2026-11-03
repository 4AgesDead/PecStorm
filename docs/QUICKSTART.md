# Быстрый старт: как работать с проектом

Короткая инструкция «что запускать и что смотреть». Полные детали —
[docs/TRUCK_DATASET.md](TRUCK_DATASET.md), числа прогона — [docs/RESULTS.md](RESULTS.md),
открытые вопросы организаторам — [docs/OPEN_QUESTIONS.md](OPEN_QUESTIONS.md).

## 0. Окружение

* Python: запускать через лаунчер **`py`** — он указывает на
  `C:\Users\BeGraphics\AppData\Local\Programs\Python\Python314\python.exe`
  (проверено: `py -V` → 3.14.10; `py -0p` показывает этот путь как основной).
  Явный вариант — `py -3.14`. Команду `python` не использовать: на PATH стоит
  заглушка Microsoft Store (`WindowsApps\python.exe`), она падает.
* Зависимости: `py -m pip install -r requirements.txt` (пины проверены: numpy 2.5.4,
  opencv 5.0.0, scipy 1.18.1, matplotlib 3.11.2, ultralytics 8.4.201, torch 2.15.0).
* Веса `yolov8x.pt` / `yolov8n.pt` лежат в корне и в git не входят
  (`.gitignore`); при отсутствии ultralytics скачает их сам.
* Все команды — из корня репозитория (`C:\Users\BeGraphics\hahaton-1`).

Проверка, что ядро живо (≈3 с):

```powershell
py -m unittest discover -s tests     # сейчас 70 тестов, OK
```

## 1. Данные

```
huniashahatona/zadania/camera/        5 MP4 (1920×1080, 30 fps, кадры 0..1000)
huniashahatona/zadania/xyzpowcameta/  5 camera CSV (CP1251) + GT трека
datasets/truck/                       manifest, калибровка якоря, sidecar-репары
```

Исходные файлы не изменяются: manifest только читает их и считает sha256.

## 2. Полный цикл (5 стадий + отчёт)

```powershell
# 1. Инвентаризация: manifest, проверка кадров/CSV/GT (секунды)
py apps/import_truck_dataset.py

# 2. Детекция YOLO по всем кадрам. ДОЛГО: ~53 мин CPU для 1001 кадра × 5 камер
py apps/detect_truck.py --start 0 --end 1000 --step 1 `
    --model yolov8x.pt --conf 0.25 --imgsz 960 `
    --out out/truck/detections.csv --stats out/truck/detect_stats.json

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

## 5. Если данных станет больше / другая сцена

1. Положить новые MP4 и camera CSV в те же подпапки `huniashahatona/zadania/`.
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
