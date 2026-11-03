"""Live-режим: камеры (файлы/RTSP/синтетика) -> 3D-позиции в консоли, запись треков.

Запуск:
    python -m apps.live --config config/demo.json --max-frames 150 --record out/live.csv

Конфиг может описывать синтетику (секция ``sim``) или реальные потоки
(``calibration`` + ``streams`` c ``kind``: ``file``/``stream``). Завершение:
по ``--max-frames`` или по стоп-защите ``stall_seconds`` (источники исчерпаны).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from multiview3d.pipeline import MultiViewPipeline, PipelineConfig
from multiview3d.visualize import tracks_to_csv


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Live-конвейер multiview3d")
    ap.add_argument("--config", default="config/demo.json")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--record", default=None, help="Записать треки в CSV")
    ap.add_argument("--quiet", action="store_true", help="Не печатать позиции каждого шага")
    args = ap.parse_args(argv)

    config = PipelineConfig.from_json(args.config)
    pipe = MultiViewPipeline(config.cameras, config)
    sources, dataset = pipe.build_sources()

    tracks = pipe.run(sources, max_frames=args.max_frames)
    st = pipe.stats()
    if not args.quiet:
        for tr in pipe.smoothed_tracks():
            pts = tr.points
            if pts:
                last = pts[-1]
                print(f"трек {tr.track_id}: {len(pts)} точек, "
                      f"последняя ({last.position[0]:.2f}, {last.position[1]:.2f}, "
                      f"{last.position[2]:.2f}) м @ t={last.timestamp:.2f}")
    print(f"Обработано кадров: {st['frames']}; частота обработки {st['processing_fps']:.1f} fps; "
          f"шаг {st['step_ms_mean']:.1f} мс (max {st['step_ms_max']:.1f})")
    if "readers" in st:
        for name, rs in st["readers"].items():
            print(f"читатель {name}: прочитано {rs['frames_read']}, потеряно {rs['frames_dropped']}, "
                  f"средняя задержка {rs['mean_latency_ms']:.2f} мс")

    if args.record:
        tracks_to_csv(pipe.smoothed_tracks(), args.record)
        print(f"Треки записаны: {args.record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
