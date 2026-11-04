"""График X-Y траектории scene2: измерения и интерполяция."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="График траектории scene2")
    ap.add_argument("--submission", default="results/scene2/submission_scene2.csv")
    ap.add_argument("--out", default="results/scene2/scene2_submission.png")
    args = ap.parse_args(argv)

    with Path(args.submission).open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    xs = [float(r["x"]) for r in rows]
    ys = [float(r["y"]) for r in rows]
    mx = [float(r["x"]) for r in rows if r["status"] == "MEASURED"]
    my = [float(r["y"]) for r in rows if r["status"] == "MEASURED"]

    measured = sum(1 for r in rows if r["status"] == "MEASURED")
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.plot(xs, ys, "-", color="#1b7f3b", lw=1.2, label="траектория (PCHIP + сглаживание)")
    ax.scatter(mx, my, s=9, c="#d17b00", label=f"MEASURED ({measured})", zorder=3)
    ax.scatter(xs[0], ys[0], s=70, facecolors="none", edgecolors="k",
               label=f"старт, кадр {rows[0]['frame']}")
    ax.scatter(xs[-1], ys[-1], s=70, marker="s", facecolors="none", edgecolors="r",
               label=f"финиш, кадр {rows[-1]['frame']}")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel("X, м")
    ax.set_ylabel("Y, м")
    ax.set_title("scene2: траектория грузовика, 1932 кадра")
    ax.legend(loc="best", fontsize=9)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"сохранено: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
