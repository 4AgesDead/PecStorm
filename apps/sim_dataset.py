"""Офлайн-прогон синтетической сцены: конвейер -> треки -> метрики -> файлы.

Запуск:
    python -m apps.sim_dataset --config config/demo.json --out out/ [--frames N]
                               [--detector blob|oracle] [--no-smooth] [--no-plot]

Выход: ``tracks.csv``, ``tracks.json``, ``truth.csv``, ``metrics.json``,
``trajectories.png`` (если matplotlib доступен).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from multiview3d.metrics import evaluate
from multiview3d.pipeline import MultiViewPipeline, PipelineConfig
from multiview3d.visualize import plot_tracks_3d, tracks_to_csv, tracks_to_json, truth_to_csv


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Синтетический прогон конвейера multiview3d")
    ap.add_argument("--config", default="config/demo.json")
    ap.add_argument("--out", default="out")
    ap.add_argument("--frames", type=int, default=None, help="Ограничить число кадров")
    ap.add_argument("--detector", choices=["blob", "oracle"], default=None,
                    help="Переопределить детектор из конфига")
    ap.add_argument("--no-smooth", action="store_true", help="Не применять RTS-сглаживание")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    config = PipelineConfig.from_json(args.config)
    if args.detector is not None:
        config.detector = {"kind": args.detector,
                           **({"colors": []} if args.detector == "blob" else {})}
    pipe = MultiViewPipeline(config.cameras, config)
    sources, dataset = pipe.build_sources(repeat=False)
    if dataset is None:
        print("Ошибка: конфиг без секции sim — для этого приложения нужна синтетика", file=sys.stderr)
        return 2

    tracks_raw = pipe.run_offline(sources, max_frames=args.frames)
    tracks = tracks_raw if args.no_smooth else pipe.smoothed_tracks()
    truth = [[p(t) for t in dataset.frame_times] for p in dataset.paths]

    rep = evaluate(tracks, truth, dataset.labels, fps=dataset.fps,
                   unit=float(dataset.cameras[0].unit), pipeline_stats=pipe.stats())

    tracks_to_csv(tracks, out / "tracks.csv")
    tracks_to_json(tracks, out / "tracks.json",
                   meta={"config": args.config, "detector": config.detector,
                         "smoothing": not args.no_smooth, "stats": pipe.stats()})
    truth_to_csv(truth, dataset.labels, dataset.frame_times, out / "truth.csv")
    (out / "metrics.json").write_text(
        json.dumps(rep.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    if not args.no_plot:
        try:
            plot_tracks_3d(tracks, truth, path=out / "trajectories.png",
                           title=f"multiview3d: {config.detector.get('kind', '?')}")
        except Exception as exc:  # matplotlib может отсутствовать
            print(f"Предупреждение: график не сохранён ({type(exc).__name__}: {exc})")

    print(f"Кадров обработано: {pipe.stats()['frames']}, треков: {len(tracks)}")
    print(f"Сопоставлено: {rep.pairs}")
    print(f"Ошибка 3D (мм): среднее={rep.overall.mean_mm}, max={rep.overall.max_mm}, "
          f"p95={rep.overall.p95_mm}, точек={rep.overall.n_points}")
    print(f"Полнота: {rep.completeness}; ID-switch: {rep.id_switches}")
    print(f"Частота обработки: {rep.processing_fps:.1f} fps; шаг {rep.step_ms_mean:.1f} мс")
    print(f"Файлы: {out / 'tracks.csv'}, {out / 'tracks.json'}, {out / 'truth.csv'}, "
          f"{out / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
