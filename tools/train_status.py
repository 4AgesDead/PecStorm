"""Статус обучения ultralytics: сколько эпох прошло, темп, ETA, лучший чекпойнт.

Читает ``runs/.../results.csv`` (по строке на завершённую эпоху) и печатает:
номер последней эпохи, время на эпоху, оценку до конца, лучшую эпоху по
mAP50-95 и когда обновлялись веса.

Запуск:
    python tools/train_status.py                       # самый свежий прогон
    python tools/train_status.py --run runs/detect/out/train/scene_v2_m --total 60
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_ROOT = Path("runs/detect/out/train")


def newest_run(root: Path) -> Path | None:
    if not root.exists():
        return None
    runs = [p for p in root.iterdir() if (p / "results.csv").exists()]
    if not runs:
        return None
    return max(runs, key=lambda p: (p / "results.csv").stat().st_mtime)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Статус обучения YOLO")
    ap.add_argument("--run", default=None, help="каталог прогона (по умолчанию новейший)")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--total", type=int, default=None, help="план эпох (для ETA)")
    ap.add_argument("--last", type=int, default=3, help="сколько последних эпох печатать")
    args = ap.parse_args(argv)

    run = Path(args.run) if args.run else newest_run(Path(args.root))
    if run is None or not (run / "results.csv").exists():
        print("нет results.csv: обучение ещё не завершило первую эпоху "
              "(или указан неверный --run)", file=sys.stderr)
        return 2

    with (run / "results.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print(f"{run}: results.csv пуст — первая эпоха ещё идёт", file=sys.stderr)
        return 2

    def f(row, key):
        try:
            return float(row[key])
        except (KeyError, ValueError):
            return float("nan")

    times = [f(r, "time") for r in rows]
    done = len(rows)
    last_epoch = int(f(rows[-1], "epoch"))
    epoch_sec = (times[-1] - times[-6]) / 5 if len(times) >= 6 else (
        times[-1] / max(last_epoch, 1))
    best = max(rows, key=lambda r: f(r, "metrics/mAP50-95(B)"))

    print(f"прогон: {run}")
    print(f"эпох завершено: {done} (последний номер {last_epoch}), "
          f"время последней эпохи ≈{epoch_sec:.0f} с")
    print(f"лучшая эпоха по mAP50-95: {int(f(best, 'epoch'))} — "
          f"P={f(best, 'metrics/precision(B)'):.3f}, R={f(best, 'metrics/recall(B)'):.3f}, "
          f"mAP50={f(best, 'metrics/mAP50(B)'):.3f}, "
          f"mAP50-95={f(best, 'metrics/mAP50-95(B)'):.3f}")
    print(f"последние {min(args.last, done)} эпох:")
    for r in rows[-args.last:]:
        print(f"  эпоха {int(f(r, 'epoch')):3d}: "
              f"P={f(r, 'metrics/precision(B)'):.3f} "
              f"R={f(r, 'metrics/recall(B)'):.3f} "
              f"mAP50={f(r, 'metrics/mAP50(B)'):.3f} "
              f"mAP50-95={f(r, 'metrics/mAP50-95(B)'):.3f} "
              f"({f(r, 'time'):.0f} с суммарно)")

    if args.total:
        left = max(args.total - done, 0)
        eta_min = left * epoch_sec / 60.0
        print(f"осталось эпох: {left} → ≈{eta_min:.0f} мин "
              f"(≈{eta_min / 60:.1f} ч) при текущем темпе")

    for name in ("best.pt", "last.pt"):
        p = run / "weights" / name
        if p.exists():
            ts = p.stat().st_mtime
            import datetime
            print(f"{name}: обновлён {datetime.datetime.fromtimestamp(ts):%H:%M:%S}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
