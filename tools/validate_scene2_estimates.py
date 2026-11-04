"""Проверка оценок scene2 на физическую правдоподобность (без GT).

Что проверяем:
* сколько кадров MEASURED и какие камеры участвуют;
* скорость между соседними MEASURED-кадрами (скачки = вероятные ложные треки);
* расхождение с предыдущей версией оценки там, где обе измерены.

Это self-consistency, а не MAE: GT в scene2 нет.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np


def load(path: Path) -> dict[int, dict]:
    with path.open(encoding="utf-8") as fh:
        return {int(r["frame"]): r for r in csv.DictReader(fh)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Проверка оценок scene2")
    ap.add_argument("--estimates", default="out/scene2/est_scene2_fixed.csv")
    ap.add_argument("--baseline", default="out/scene2/est_scene2_step1.csv")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--max-speed", type=float, default=25.0,
                    help="порог правдоподобия, м/с")
    ap.add_argument("--min-gap", type=int, default=5,
                    help="считать скорость только по парам с разницей >= N кадров "
                         "(меньше — это дрожание оценок, а не движение)")
    args = ap.parse_args(argv)

    est = load(Path(args.estimates))
    rows = [est[f] for f in sorted(est) if est[f].get("status") == "MEASURED"]
    print(f"кадров в файле: {len(est)}, MEASURED: {len(rows)}")
    if not rows:
        return 2
    cams = Counter()
    for r in rows:
        for v in (r.get("views") or "").split("|"):
            if v:
                cams[v] += 1
    print("камеры в измерениях:", dict(sorted(cams.items())))
    pairs = Counter()
    for r in rows:
        vs = tuple(sorted(v for v in (r.get("views") or "").split("|") if v))
        pairs[vs] += 1
    print("топ наборов видов:")
    for k, n in pairs.most_common(8):
        print(f"  {'+'.join(k) or '—'}: {n}")

    fps = args.fps
    speeds = []
    jitter = []
    prev = None
    bad = []
    for r in rows:
        f = int(r["frame"])
        p = np.array([float(r["x"]), float(r["y"]), float(r["z"])])
        if prev is not None and f > prev[0]:
            step = float(np.linalg.norm(p - prev[1]))
            if f - prev[0] < args.min_gap:
                jitter.append(step)
            else:
                v = step / ((f - prev[0]) / fps)
                speeds.append(v)
                if v > args.max_speed:
                    bad.append((prev[0], f, round(v, 1)))
        prev = (f, p)
    if jitter:
        j = np.asarray(jitter)
        print(f"дрожание между соседними кадрами: медиана {np.median(j):.2f} м, "
              f"P95 {np.percentile(j, 95):.2f} м (n={j.size})")
    if speeds:
        speeds = np.asarray(speeds)
        print(f"скорость между соседними MEASURED: медиана {np.median(speeds):.1f} м/с, "
              f"P95 {np.percentile(speeds, 95):.1f}, max {speeds.max():.1f}")
        print(f"скачков > {args.max_speed:.0f} м/с: {len(bad)}")
        for b in bad[:10]:
            print("  ", b)

    errs = [float(r.get("mean_reproj_px") or "nan") for r in rows]
    errs = np.asarray([e for e in errs if np.isfinite(e)])
    if errs.size:
        print(f"невязка репроекции: медиана {np.median(errs):.1f} px, "
              f"P95 {np.percentile(errs, 95):.1f} px")

    base_path = Path(args.baseline)
    if base_path.exists():
        base = load(base_path)
        common = [(f, np.array([float(base[f]["x"]), float(base[f]["y"]), float(base[f]["z"])]),
                   np.array([float(est[f]["x"]), float(est[f]["y"]), float(est[f]["z"])]))
                  for f in sorted(est)
                  if f in base and base[f].get("status") == "MEASURED"
                  and est[f].get("status") == "MEASURED"]
        if common:
            d = np.array([float(np.linalg.norm(a - b)) for _, a, b in common])
            print(f"общих MEASURED с базой: {len(common)}; расхождение "
                  f"медиана {np.median(d):.2f} м, P90 {np.percentile(d, 90):.2f} м, "
                  f"max {d.max():.2f} м")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
