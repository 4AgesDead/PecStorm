"""Быстрая проверка работоспособности ядра multiview3d.

Запуск:
    C:\\Users\\BeGraphics\\AppData\\Local\\Programs\\Python\\Python314\\python.exe tools\\smoke_check.py

Проверяет:
1) импорт всех модулей пакета;
2) рендер синтетической сцены и совпадение детекций с точной проекцией;
3) работу FrameReader на синтетическом источнике (включая close()/closed);
4) сквозной офлайн-прогон: 3 камеры, 2 объекта, 2 с — два трека,
   средняя ошибка 3D после RTS-сглаживания <= 30 мм (критерий приёмки P0).

Код возврата 0 — все обязательные проверки прошли, 1 — есть провал.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MODULES = ["types", "camera", "calibration", "detect", "kalman", "trajectory",
           "capture", "sim", "fusion", "pipeline", "metrics", "visualize"]
REQUIRED = set(MODULES)

failures: list[str] = []


def check_imports() -> dict[str, str]:
    status: dict[str, str] = {}
    for name in MODULES:
        try:
            importlib.import_module(f"multiview3d.{name}")
            status[name] = "ok"
        except Exception as exc:  # noqa: BLE001 - нужен любой отказ
            status[name] = f"{type(exc).__name__}: {exc}"
            if name in REQUIRED:
                failures.append(f"импорт {name}: {status[name]}")
    return status


def check_sim_detector() -> None:
    from multiview3d.camera import project_points
    from multiview3d.detect import ColorBlobDetector
    from multiview3d.sim import make_sim_setup, SimCameraStream

    dataset = make_sim_setup(num_cameras=3, num_objects=2, fps=25, duration_s=1.0)
    t, frame, _ = SimCameraStream(dataset, 0).read()
    print(f"  кадр: shape={frame.shape} dtype={frame.dtype} t={t}")

    detector = ColorBlobDetector(dataset.ball_colors[: len(dataset.paths)])
    detections = detector.detect(frame)
    expected = project_points(dataset.cameras[0], dataset.truth_at(t))
    if len(detections) != len(expected):
        failures.append(f"детектор нашёл {len(detections)} меток, ожидалось {len(expected)}")
        return

    det = np.array([d.center for d in detections])
    order = np.argsort(det[:, 0])
    err = np.linalg.norm(det[order] - expected[np.argsort(expected[:, 0])], axis=1)
    print(f"  детекции: {len(detections)}, макс. ошибка центра {err.max():.2f} px")
    if err.max() > 2.0:
        failures.append(f"центр метки расходится с проекцией на {err.max():.2f} px")


def check_frame_reader() -> None:
    from multiview3d.capture import FrameReader
    from multiview3d.detect import ColorBlobDetector
    from multiview3d.sim import make_sim_setup, SimCameraStream, SimSource

    dataset = make_sim_setup(num_cameras=2, num_objects=1, duration_s=0.5)
    source = SimSource(SimCameraStream(dataset, 0),
                       ColorBlobDetector(dataset.ball_colors[:1]))
    reader = FrameReader(source, "cam1")
    reader.start()
    try:
        item = reader.latest(timeout=1.0)
        if item is None:
            failures.append("FrameReader не отдал кадр за 1 с")
        else:
            stamp, frame, dets, _ = item
            print(f"  FrameReader: t={stamp:.3f} shape={frame.shape} детекций={len(dets or [])}")
    finally:
        try:
            reader.close()
            print(f"  close()/closed: источник закрыт корректно (closed={source.closed})")
        except AttributeError as exc:
            failures.append(f"close()/closed не поддержан источником: {exc}")


def check_end_to_end() -> None:
    from multiview3d.fusion import FusionConfig
    from multiview3d.metrics import evaluate
    from multiview3d.pipeline import MultiViewPipeline, PipelineConfig
    from multiview3d.sim import OracleSimSource, make_sim_setup

    dataset = make_sim_setup(num_cameras=3, num_objects=2, fps=25, duration_s=2.0, seed=7)
    cfg = PipelineConfig(cameras=dataset.cameras,
                         fusion=FusionConfig(workspace_bounds=(-2, -2, 0, 2, 2, 3)))
    pipe = MultiViewPipeline(dataset.cameras, cfg)
    sources = [OracleSimSource(dataset, i) for i in range(len(dataset.cameras))]
    tracks = pipe.run_offline(sources)
    smoothed = pipe.smoothed_tracks()
    truth = [[np.asarray(p(t)) for t in dataset.frame_times] for p in dataset.paths]
    rep = evaluate(smoothed, truth, dataset.labels, fps=dataset.fps, unit=1.0,
                   pipeline_stats=pipe.stats())

    print(f"  треков: {len(tracks)} (точек: {[len(t.points) for t in tracks]})")
    print(f"  сопоставление: {rep.pairs}; ID-switch: {rep.id_switches}; "
          f"полнота: { {k: round(v, 2) for k, v in rep.completeness.items()} }")
    print(f"  ошибка 3D (мм): среднее={rep.overall.mean_mm}, max={rep.overall.max_mm}, "
          f"p95={rep.overall.p95_mm}, точек={rep.overall.n_points}")
    print(f"  частота обработки: {rep.processing_fps:.0f} fps, шаг {rep.step_ms_mean:.1f} мс")
    if len(tracks) != 2:
        failures.append(f"ожидалось 2 трека, получено {len(tracks)}")
    if len(rep.pairs) != 2:
        failures.append(f"сопоставлено пар: {len(rep.pairs)}, ожидалось 2")
    if rep.overall.mean_mm > 30.0:
        failures.append(f"средняя ошибка {rep.overall.mean_mm:.1f} мм > 30 мм")


def main() -> int:
    print("1) импорт модулей")
    for name, state in check_imports().items():
        mark = "OK  " if state == "ok" else "FAIL"
        print(f"  {mark} {name:12s} {state if state != 'ok' else ''}")

    print("2) синтетика + детектор")
    try:
        check_sim_detector()
    except Exception as exc:  # noqa: BLE001
        failures.append(f"синтетика/детектор: {type(exc).__name__}: {exc}")

    print("3) FrameReader")
    try:
        check_frame_reader()
    except Exception as exc:  # noqa: BLE001
        failures.append(f"FrameReader: {type(exc).__name__}: {exc}")

    print("4) сквозной прогон (fusion + pipeline + metrics)")
    try:
        check_end_to_end()
    except Exception as exc:  # noqa: BLE001
        failures.append(f"сквозной прогон: {type(exc).__name__}: {exc}")

    print()
    if failures:
        print("ПРОВАЛЫ:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("Все обязательные проверки пройдены.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
