# Датасет «Пеленга» (truck): формат, конверсия, ограничения

Документ фиксирует **проверенные факты** о выданном наборе и все допущения,
которые влияют на метрическую точность. Всё, что не подтверждено
организаторами, помечено как допущение (список вопросов —
[docs/OPEN_QUESTIONS.md](OPEN_QUESTIONS.md)).

## 1. Файлы набора

Корень: `huniashahatona/zadania/` (не изменяется; только чтение).

| Что | Путь | Факт |
|---|---|---|
| видео камер | `camera/Way1.FinalImage.CineCameraActor{1..5}.mp4` | 1920×1080, `CAP_PROP_FPS = 30`, 1001 кадр (индексы 0..1000) |
| параметры камер | `xyzpowcameta/object_transform_CineCameraActor{1..5}.csv` | CP1251, `;`, 1002 строки с `Frame` 0..1001 |
| GT цели | `xyzpowcameta/object_transform_SM_AA_track.csv` | CP1251, `;`, 1000 валидных строк (кадры 0..1001 без 402 и 691) |

Проверка: `python apps/import_truck_dataset.py --no-hashes`
(manifest — `datasets/truck/dataset_manifest.json`). Цифры из manifest:
`fps=30`, `width=1920`, `height=1080`, `common_frame_range=[0, 1000]`,
`csv_rows_without_video = 1` для каждой камеры.

## 2. Формат camera CSV

```
Frame;Loc_X;Loc_Y;Loc_Z;Rot_Roll;Rot_Pitch;Rot_Yaw;;;Focal length;20
0;9799.8057;-3900.0;1487.5303;-4.4117;-6.9741;-164.0;;;матрица;23,76*13,365мм
1;...
```

* строка заголовка содержит `Focal length` = 20 (мм) в последней ячейке;
* в строке кадра 0 в тех же ячейках лежит `матрица;23,76*13,365мм` — это
  **данные кадра 0 с метаданными**, а не отдельная строка заголовка
  (парсер это учитывает: `unreal_camera.parse_camera_csv`);
* координаты — сантиметры Unreal world, углы — градусы FRotator
  (roll, pitch, yaw).

## 3. Конверсия в метрическую модель OpenCV

Реализация: `multiview3d/unreal_camera.py`.

1. Единицы: `м = см * 0.01` (`CM_TO_M = 0.01`).
2. Мир: правая система `X_u = X_ue, Y_u = -Y_ue, Z_u = Z_ue`
   (`unreal_to_right`, `S_FLIP = diag(1, -1, 1)`, det = −1).
3. Оси камеры: UE `X` вперёд, `Y` вправо, `Z` вверх → OpenCV `x` вправо,
   `y` вниз, `z` вперёд.
4. `R_w2c = M_s @ R_ue^T`, `t = -R_w2c @ C_w`, где `C_w` — центр камеры в
   метрах; `det(R_w2c) = +1` (проверяется тестом).
5. **Эмпирическая поправка знака pitch**: в CSV `Rot_Pitch` входит с
   обратным знаком относительно рендера (`ue_rot_to_matrix(roll, -pitch, yaw)`).
   Поправка получена фитом позы по точкам касания колёс
   (`tools/pose_fit.py`), организаторами не подтверждена.
6. Intrinsics: `fx = focal_mm * width / sensor_w`,
   `fy = focal_mm * height / sensor_h`, principal point в центре кадра,
   дисторсия нулевая. Для 1920×1080 и 20 мм / 23.76×13.365 мм:
   `fx = 20*1920/23.76 ≈ 1616.16`, `fy = 20*1080/13.365 ≈ 1616.16`,
   `cx = 960`, `cy = 540`.

Проверки: `tests/test_unreal_ingest.py` (единицы, оси, статичность камер,
непрерывность yaw cam5, общая область кадров) и
`python experiments/visibility_analysis.py`.

## 4. Камеры и синхронизация

* cam1–cam4 статичны (все строки CSV имеют одинаковые `loc` и `rot`) —
  `CameraStateProvider.is_static == True`, тест
  `TestStaticCameraPose.test_cameras_1_4_static`.
* cam5 вращается по yaw; CSV содержит yaw в диапазоне ±180° (wrap), поэтому
  поза строится из угла кадра напрямую, без линейной интерполяции углов.
  Максимальный шаг между соседними кадрами < 5° (тест
  `TestCamera5YawUnwrap.test_rotation_continuity`).
* **PTS контейнера OpenCV не отдаёт.** Синхронизация — по номеру кадра
  (`timestamp = frame / fps`), что является допущением: гарантий
  синхронности от организаторов нет (вопрос №6).

### Видимость цели (факт, `experiments/visibility_analysis.py --step 5`)

Проекция GT-пивота в кадр по 201 кадру (шаг 5):

| Число камер с пивотом в кадре | Кадров | Доля |
|---|---|---|
| 1 | 6 | 3.0 % |
| 2 | 13 | 6.5 % |
| 3 | 107 | 53.2 % |
| 4 | 75 | 37.3 % |

Итого ≥2 видов: 195 из 201 кадров (97.0 %). По камерам: cam1 — 15.9 %,
cam2 — 76.6 %, cam3 — 55.2 %, cam4 — 77.1 %, cam5 — 100 % кадров.

Это **верхняя** оценка: она не учитывает, найдёт ли детектор объект
(малый размер, окклюзия).

## 5. Ground truth и его дефекты

`parse_track_csv` не приводит «авг.13» и «июл.25» к числам, а складывает в
`invalid_cells`; такие кадры **исключаются** из GT:

| Кадр | Строка | Колонка | Сырое значение |
|---|---|---|---|
| 402 | 404 | `Loc_X` | `авг.13` |
| 691 | 693 | `Loc_Y` | `июл.25` |

Sidecar `datasets/truck/ground_truth_repairs.json` содержит гипотезу
восстановления (`8.13` и `7.25`, формат mm.dd) с
`verified_by_organizer = false`; по умолчанию **не применяется**. Включить
можно только явным флагом `apps/evaluate_truck.py
--accept-unverified-repairs`. Исходный CSV не изменяется.

## 6. Ключевая проблема точности: 2D-якорь ≠ pivot

Детектор возвращает bbox и его центр. GT — это `pivot` актора
(`SM_AA_track`). Точка «центр bbox» не соответствует одной и той же
3D-точке в разных видах: при разных ракурсах это разные точки поверхности
груза/кабины. Следствия:

* триангуляция bbox-центров даёт систематически смещённую 3D-точку;
* средняя репроекционная невязка такой триангуляции составляет десятки
  пикселей (см. `experiments/projection_sanity.py`), поэтому порог
  `max_reprojection_px` для truck-workflow поднят до 30 px (в синтетике
  остаётся 4 px);
* введён **постоянный калибруемый 3D-сдвиг якоря**
  `FusionConfig.anchor_offset_m`, оцениваемый на обучающих кадрах
  (`tools/anchor_fit_yolo.py`), а не на проверочных.

Модель сдвига: `project(cam_f, pivot_gt[f] + delta) ≈ bbox_center`.
`delta` — три числа в метрах, одни на все камеры и кадры. Остаточная
невязка после фита отражает не «шум детектора», а несовершенство модели
(параллакс разных точек объекта, ошибки intrinsics/extrinsics, неизвестный
anchor).

Актуальные числа фита и проверки на отложенных кадрах — в
`datasets/truck/anchor_offset.json` и `docs/RESULTS.md`.

## 6a. Второй якорь: 2D-сдвиг по камерам

Постоянный 3D-сдвиг компенсирует только **общее** смещение. Проверка
`leave-one-camera-out` в `tools/anchor_fit_yolo.py` показала, что сдвиг,
оценённый без одной камеры, на этой камере почти не уменьшает невязку
(числа — в `docs/RESULTS.md`): у каждого вида своё смещение «bbox ↔ pivot».
Поэтому введён второй, независимый параметр:

* `FusionConfig.anchor_px_offset: dict[str, (du, dv)]` — сдвиг в пикселях
  **на камеру**;
* оценивается как медиана `bbox_center - project(pivot_gt)` на train-кадрах
  (`tools/anchor_fit_px.py`), на holdout только проверяется;
* применяется **вычитанием** из центра детекции до триангуляции:
  `px_corrected = px - offset`.

Оба якоря опциональны: если JSON нет, сдвиг = 0 и в лог печатается
предупреждение. Измеренные результаты (полный прогон, кадры 0..1000):
без сдвига MAE 11.89 м, общий 3D-сдвиг 7.40 м, 2D-сдвиг по камерам 10.00 м,
сдвиг по наборам видов 4.28 м (in-sample), parity-CV 4.51 м для
`cam3|cam5` и `cam4|cam5`. Поэтому основным выбран сдвиг по наборам видов;
детали и оговорки — в `docs/RESULTS.md` §5.

## 7. Детектор

* ML, ultralytics YOLO, класс COCO `truck` (дополнительно принимаются
  `car`, `bus`, `train`: на дальних планах грузовик классифицируется как
  `car`/`truck` неустойчиво).
* Рабочая конфигурация: `yolov8x.pt`, `imgsz=960`, `conf=0.25`, CPU.
  `yolov8n` на этом наборе почти не находит объект (см.
  `out/experiments/detector_benchmark.json` после прогона
  `experiments/detector_benchmark.py`).
* Фоновое вычитание как основной детектор **отклонено**: на статичных
  камерах даёт маску дороги/облаков, а не объект.

Модули: `multiview3d/detect.py` (`YoloDetector`),
`multiview3d/sources.py` (`DetectingFrameSource` — детектор поверх любого
источника кадров).

## 8. Workflow (полный цикл)

```powershell
# 1. инвентаризация и manifest
py apps/import_truck_dataset.py

# 2. детекция по всем кадрам (долго: ~0.5–0.6 с на кадр камеры на CPU)
py apps/detect_truck.py --start 0 --end 1000 --step 1 `
    --model yolov8x.pt --conf 0.25 --imgsz 960 `
    --out out/truck/detections.csv --stats out/truck/detect_stats.json

# 3. негативные варианты якоря (для сравнения; на holdout не переносятся)
py tools/anchor_fit_yolo.py --detections out/truck/detections.csv `
    --train-start 0 --train-end 500 --holdout-start 600 --holdout-end 1000 --step 1
py tools/anchor_fit_px.py --detections out/truck/detections.csv `
    --train-start 0 --train-end 500 --holdout-start 600 --holdout-end 1000

# 3a. основной вариант: 3D-сдвиг на набор видов (parity/time-CV внутри)
py tools/anchor_fit_views.py --detections out/truck/detections.csv `
    --max-reprojection-px 50

# 4. реконструкция (порог 50 px — по experiments/geometry_sweep.py)
py apps/reconstruct_truck.py --detections out/truck/detections.csv `
    --max-reprojection-px 50 `
    --out-csv out/truck/estimates.csv --out-json out/truck/estimates.json

# 5. оценка (ошибка и покрытие — раздельно)
py apps/evaluate_truck.py --estimates out/truck/estimates.csv `
    --out-json out/truck/evaluation.json --out-csv out/truck/eval_per_frame.csv

# 6. визуализация (кадры, где оценка реально есть)
py apps/visualize_truck.py --estimates out/truck/estimates.csv `
    --frames 83,190,400,600,834,924 --out out/truck/viz --video

# 7. сводка и графики (error vs frame, траектория XY, покрытие, компоненты X/Y/Z)
py tools/make_report.py

# 7a. только «итоговый» график X/Y/Z: эталон (сплошные) vs оценка (штриховые)
py tools/plot_components.py --estimates out/truck/estimates.csv `
    --out out/truck/report/components_vs_time.png `
    --error-out out/truck/report/components_error_vs_time.png
```

Эксперименты (каждый пишет JSON в `out/experiments/`):

```powershell
py experiments/visibility_analysis.py --step 5
py experiments/projection_sanity.py --detections out/truck/detections.csv
py experiments/detector_benchmark.py
py experiments/geometry_sweep.py --detections out/truck/detections.csv
py experiments/robustness_benchmark.py
```

Конфиг конвейера: `config/truck.json` (камеры, детектор, пороги слияния,
пять потоков `kind=truck` для `apps/live.py`).

## 9. Статусы и формат вывода (§11 брифа)

`Estimates3D` (`out/truck/estimates.csv` / `.json`):

* `status`: `MEASURED` (триангуляция по ≥2 видам), `PREDICTED_ONLY`
  (прогноз Калмана), `UNOBSERVABLE` (нет ни измерения, ни прогноза,
  присутствует только в отчёте оценки), `MONOCULAR` (один вид — отдельный
  счётчик покрытия);
* `method`: `triangulation` | `predicted_only` | `monocular_pnp` | `none`;
* `camera_ids`/`views`, `frame_id`, `timestamp`, `reprojection_error_px`,
  `confidence`, `position_xyz_m`, `velocity_xyz_m_s`.

`confidence` — эвристика `confidence_from_geometry` (функция репроекционной
невязки и числа видов), **не** калиброванная вероятность.

## 10. Ограничения (что нельзя утверждать)

1. Метрическая точность ограничена неизвестным anchor (вопрос №2) и
   неподтверждёнными intrinsics (вопрос №5). Сдвиг якоря — приближение.
2. Синхронность по номеру кадра не подтверждена PTS (вопрос №6).
3. На однокамерных кадрах 3D без 3D-модели объекта не восстанавливается:
   такие кадры не выдаются как измерение (только прогноз/`UNOBSERVABLE`).
4. Дисторсия не моделируется; crop/overscan/filmback не подтверждены.
5. Прогон на CPU: ~0.5–0.6 с на кадр одной камеры при `imgsz=960`
   (`yolov8x`, 1920×1080), т.е. полный набор из 5005 кадров — десятки минут.
