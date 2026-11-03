# Результаты прогона (артефакты)

Здесь лежат **зафиксированные артефакты** полного прогона truck-workflow по
датасету «Пеленга» (5 камер, кадры 0..1000, 30 fps). Сами прогоны пишутся в
`out/` (в git не входят — см. `.gitignore`), а здесь сохранена копия того, по
чему считались числа в [docs/RESULTS.md](../docs/RESULTS.md).

## Что внутри `truck/`

| файл | что это |
|---|---|
| `detections.csv` | 2D-детекции YOLO (`yolov8x.pt`, `imgsz=960`, `conf=0.25`): 1660 строк, 938 кадров |
| `detect_stats.json` | статистика детекции: время, доля кадров по камерам, итоги |
| `estimates.csv` / `.json` | 3D-оценки (DTO §11): `frame`, `x,y,z`, `vx,vy,vz`, `status`, `method`, `views`, `confidence` |
| `evaluation.json` | MAE/RMSE/median/P95/P99/max + покрытие по статусам |
| `eval_per_frame.csv` | построчно: оценка, GT, ошибка, статус по кадрам |
| `anchors_rejected/` | калибровки, которые **не переносятся** на holdout: общий 3D-сдвиг и 2D-сдвиг по камерам (негативные результаты, §5 RESULTS) |
| `experiments/` | видимость, проверка проекции bbox↔pivot, свип порогов, робастность, бенчмарк детектора, прокси-recall по пивоту (`hitrate_*.json`) |
| `report/` | графики и сводка: `components_vs_time.png` (X/Y/Z эталон vs оценка), `components_error_vs_time.png` (+ ошибка по осям), `components_with_path.png` (+ траектория в плане), `components_vs_time_6lines.png` (6 линий, как в презентации), `trajectory_xy.png`, `error_vs_frame.png`, `coverage.png`, `summary.json` |

## Ключевые числа (источник — `truck/evaluation.json`)

* покрытие: `MEASURED` 614 кадров (61.3 %), `PREDICTED_ONLY` 25 (2.5 %),
  `UNOBSERVABLE` 362 (36.2 %), `MONOCULAR` 0 %;
* MAE по 612 измеренным точкам: 4.28 м; медиана 2.91 м; P95 11.18 м;
  P99 12.63 м; max 71.84 м;
* MAE по осям: X 2.97 м, Y 2.17 м, Z 0.91 м;
* детекция: 1001 кадр, 5005 кадро-камер, 3200 с на CPU (≈0.64 с на камеру-кадр).

**Оговорка о честности:** 4.28 м — in-sample по калибровке якоря
(`datasets/truck/anchor_offset_views.json` фитился по всем кадрам). Оценка
переноса по разбиению чётные/нечётные кадры — 4.51 м для `cam3|cam5` и
`cam4|cam5`, 2.21 м для `cam1|cam2`; для 3-видных наборов калибровки нет.
Без файла калибровки MAE 11.89 м. Подробности — `docs/RESULTS.md`.

## Как воспроизвести

```powershell
py apps/import_truck_dataset.py
py apps/detect_truck.py --start 0 --end 1000 --step 1 `
    --model yolov8x.pt --conf 0.25 --imgsz 960
py tools/anchor_fit_views.py --detections out/truck/detections.csv --max-reprojection-px 50
py apps/reconstruct_truck.py --detections out/truck/detections.csv --max-reprojection-px 50
py apps/evaluate_truck.py --estimates out/truck/estimates.csv
py tools/make_report.py
```

Полный прогон детекции занимает ≈53 мин CPU; остальные стадии — секунды.
