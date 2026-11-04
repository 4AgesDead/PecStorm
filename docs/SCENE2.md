# Задание 2: scene2 (9 камер, грузовик)

## Данные

* `scene2/` — 9 MP4 `Way2.FinalImage.CineCameraActor1..9.mp4`;
* 1920×1080, 30 fps, **1932 кадра** (0..1931);
* 9 CSV с позами: `Frame;Loc_X;Loc_Y;Loc_Z;Rot_Roll;Rot_Pitch;Rot_Yaw;Focal_Length_mm`;
* сенсор `23,76 × 13,365 мм` (`scene2/matrix_size.txt`);
* **8 камер статичны, cam2 движется** (830 уникальных поз);
* **GT-файла нет** — только камеры и видео.

Цель — грузовик (та же модель, что в scene1), сдать 3D-координаты **на все
1932 кадра** в CSV. Разметки/эталона для MAE в наборе нет; оценка — скрытый GT
организаторов или self-consistency.

## Что сделано

* `multiview3d/scene2_dataset.py` — загрузчик 9 камер, парсинг фокуса/сенсора,
  движущаяся cam2, без обязательного GT.
* `apps/reconstruct_scene2.py` — детекция на 9 камерах → триангуляция →
  трекинг → CSV в схеме scene1.
* `tools/fill_scene2_trajectory.py` — заполнение всех 1932 кадров:
  PCHIP по `MEASURED`, константные края, лёгкое SG-сглаживание.

Детектор: **COCO `yolov8x`, класс `truck`**. Модель `v4b` на scene2 цель почти
не видит (переобучена на близкие ракурсы scene1); COCO находит её в
`cam3|cam7`, `cam2`, `cam6|cam9`.

## Результат

| файл | что внутри |
|---|---|
| `results/scene2/submission_scene2.csv` | 1932 строки, полная схема scene1 + статус; `x`, `y`, `z` — отдельные столбцы |
| `results/scene2/submission_scene2_xyz_final.csv` | только `frame,x,y,z` — для строгого парсера (актуальный) |
| `results/scene2/submission_scene2_xyz.csv` | старый минимальный файл; может быть занят Excel |
| `results/scene2/submission_scene2.json` | метод, диапазон, длина пути |
| `results/scene2/scene2_submission.png` | график X–Y: измерение vs интерполяция |
| `out/scene2/scene2_dashboard.mp4` | видео: 9 камер + карта + таблица (локально, не в git) |

Числа:

* **499** кадров с реальным 3D-измерением (статус `MEASURED`);
* **1433** кадра `INTERPOLATED` (PCHIP между измерениями; края — константа);
* длина пути по X–Y ≈ **457 м**;
* основные измерительные сегменты: 169..252, 376..438, 704..722,
  1054..1136, 1170..1280, 1560..1690, 1770..1822.

Схема полного CSV:

`frame,timestamp,x,y,z,vx,vy,vz,status,method,n_views,views,mean_reproj_px,confidence`

## Честно про ограничения

* GT в scene2 нет — **MAE/покрытие против эталона не посчитать**; скрытый GT
  организаторов не проверялся.
* Якоря «bbox → пивот» для scene2 не калибровались: нет GT. Координаты — сырая
  триангуляция без view/azimuth-поправок, возможен систематический сдвиг.
* **1433 строки — интерполяция, а не измерение.** На длинных пропусках
  (например, 438..704) путь может отличаться от реального.
* cam2 часто единственный вид; такие кадры не дают 3D (нужно ≥2 вида).
* `scene2/` не коммитится в git (≈1.2 ГБ); в репозитории только результаты
  и код. CSV воспроизводится командами ниже.

## Воспроизвести

```powershell
# черновик 3D по кадрам (шаг 1, COCO yolov8x)
.\.venv-rocm\Scripts\python.exe apps/reconstruct_scene2.py `
    --model yolov8x.pt --class-names truck `
    --start 0 --end 1931 --step 1 `
    --out-csv out/scene2/est_scene2_step1.csv

# заполнить все 1932 кадра
py tools/fill_scene2_trajectory.py `
    --estimates out/scene2/est_scene2_step1.csv `
    --out-csv results/scene2/submission_scene2.csv

# видео как пульт
py tools/render_scene2_video.py --out out/scene2/scene2_dashboard.mp4
```

## Что улучшать дальше

1. Ансамбль `COCO yolov8x + v1 + v2 + v4b` через `--models` (код уже есть) —
   проверить, добавляет ли он цель или только ложные парковки.
2. Для кадров с одним видом — пересечение луча с плоскостью земли `Z≈3 м`.
3. Отсев ложных треков припаркованных грузовиков по подвижности.
4. Когда появится GT — калибровка view/azimuth-якорей и честный MAE.
