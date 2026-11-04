"""Графики сравнения всех конфигураций конвейера (MAE, покрытие, recall/precision).

Читает JSON-отчёты `apps/evaluate_truck.py` и `experiments/anchor_hitrate.py`
и строит:

* `comparison_mae.png` — MAE по всем кадрам и на holdout (горизонтальные бары);
* `comparison_coverage.png` — покрытие там же;
* `comparison_tradeoff.png` — «покрытие vs MAE» (две панели: все кадры, holdout);
* `comparison_recall_precision.png` — recall/precision-прокси по детекциям;
* `comparison.json` — те же числа в машинном виде.

Запуск:
    python tools/make_comparison_charts.py [--out out/truck/report_compare]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# (подпись, eval-json, holdout-json, hitrate-json, финальная?)
CONFIGS = [
    ("база yolov8x (CPU)", "out/truck/evaluation.json", None,
     "out/experiments/hitrate_yolov8x.json", False),
    ("ONNX+DML yolov8x", "out/truck/eval_dml.json", None, None, False),
    ("v1: yolov8s, 16 эп.", "out/truck/eval_ft.json", "out/truck/eval_ft_holdout.json",
     "out/experiments/hitrate_ft.json", False),
    ("v2: yolov8m, 53 эп.", "out/truck/eval_v2m.json", "out/truck/eval_v2m_holdout.json",
     None, False),
    ("v3: + hard-negative", "out/truck/eval_v3.json", "out/truck/eval_v3_holdout.json",
     None, False),
    ("v4: m с нуля, 60 эп.", "out/truck/eval_v4.json", "out/truck/eval_v4_holdout.json",
     "out/experiments/hitrate_v4.json", False),
    ("v4b: fine-tune v4", "out/truck/eval_v4b.json", "out/truck/eval_v4b_holdout.json",
     "out/experiments/hitrate_v4b.json", False),
    ("v1+v2 + азимут", "out/truck/eval_az_final.json", "out/truck/eval_az2_holdout.json",
     "out/experiments/hitrate_ens.json", False),
    ("v1+v2+v4 + азимут", "out/truck/eval_ens_v4v2v1.json",
     "out/truck/eval_ens_v4v2v1_holdout.json",
     "out/experiments/hitrate_ens_v4v2v1.json", False),
    ("v1+v2+v4b + азимут", "out/truck/eval_ens_v4b_v2v1.json",
     "out/truck/eval_ens_v4b_v2v1_holdout.json",
     "out/experiments/hitrate_ens_v4b_v2v1.json", True),
    ("все четыре + азимут", "out/truck/eval_ens_all4.json",
     "out/truck/eval_ens_all4_holdout.json",
     "out/experiments/hitrate_ens_all4.json", False),
]


def load(path: str | None) -> dict | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Сравнительные графики конфигураций")
    ap.add_argument("--out", default="out/truck/report_compare")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for title, ev, hold, hit, final in CONFIGS:
        d = load(ev)
        if d is None:
            print(f"пропуск (нет {ev}): {title}")
            continue
        m, c = d["measured"], d["coverage"]
        h = load(hold)
        hi = load(hit)
        rows.append({
            "title": title,
            "final": final,
            "n": m["n_points"],
            "mae": m["mae_mm"] / 1000,
            "median": m["median_mm"] / 1000,
            "p95": m["p95_mm"] / 1000,
            "coverage": c["measured"],
            "holdout_mae": (h["measured"]["mae_mm"] / 1000) if h else None,
            "holdout_coverage": (h["coverage"]["measured"]) if h else None,
            "recall": (hi["totals"]["recall_proxy"]) if hi else None,
            "precision": (hi["totals"]["precision_proxy"]) if hi else None,
        })

    (out / "comparison.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    order = sorted(rows, key=lambda r: r["mae"])
    labels = [r["title"] for r in order]
    y = np.arange(len(order))
    colors = ["#1b7f3b" if r["final"] else "#4c72b0" for r in order]

    # --- MAE ---
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(order) + 1.6))
    vals = [r["mae"] for r in order]
    ax.barh(y, vals, color=colors, alpha=0.9)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    for i, r in enumerate(order):
        txt = f"{r['mae']:.2f} м (n={r['n']})"
        if r["holdout_mae"] is not None:
            txt += f"; holdout {r['holdout_mae']:.2f}"
        ax.text(r["mae"] + 0.05, i, txt, va="center", fontsize=8)
    ax.set_xlabel("MAE по MEASURED, метры (меньше — лучше)")
    ax.set_title("Сравнение конфигураций: ошибка 3D")
    ax.set_xlim(0, max(vals) * 1.35)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "comparison_mae.png", dpi=140)
    plt.close(fig)

    # --- покрытие ---
    order_c = sorted(rows, key=lambda r: r["coverage"])
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(order_c) + 1.6))
    vals = [r["coverage"] * 100 for r in order_c]
    ax.barh(np.arange(len(order_c)), vals,
            color=["#1b7f3b" if r["final"] else "#dd8452" for r in order_c])
    ax.set_yticks(np.arange(len(order_c)), [r["title"] for r in order_c])
    ax.invert_yaxis()
    for i, r in enumerate(order_c):
        txt = f"{r['coverage'] * 100:.1f} %"
        if r["holdout_coverage"] is not None:
            txt += f"; holdout {r['holdout_coverage'] * 100:.1f} %"
        ax.text(r["coverage"] * 100 + 0.4, i, txt, va="center", fontsize=8)
    ax.set_xlabel("доля кадров с измерением, % (больше — лучше)")
    ax.set_title("Сравнение конфигураций: покрытие")
    ax.set_xlim(0, 100)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "comparison_coverage.png", dpi=140)
    plt.close(fig)

    # --- trade-off ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, mode in zip(axes, ("all", "holdout")):
        pts = []
        for r in rows:
            cx = r["coverage"] * 100 if mode == "all" else (
                (r["holdout_coverage"] or 0) * 100)
            cy = r["mae"] if mode == "all" else (r["holdout_mae"] or 0)
            if not cx or not cy:
                continue
            pts.append((cx, cy, r))
        pts.sort(key=lambda t: (t[0], t[1]))
        for i, (cx, cy, r) in enumerate(pts):
            ax.scatter(cx, cy, s=90, color="#1b7f3b" if r["final"] else "#4c72b0",
                       edgecolor="black", linewidth=0.4, zorder=3)
            dy = 7 if i % 2 == 0 else -12
            ax.annotate(r["title"], (cx, cy), textcoords="offset points",
                        xytext=(6, dy), fontsize=6.5)
        ax.set_xlabel("покрытие, %")
        ax.set_title("все кадры" if mode == "all" else "holdout 700..1000")
        ax.grid(alpha=0.3)
        ax.axvline(90, color="red", linestyle="--", linewidth=1, alpha=0.6)
        ax.text(0.905, 0.02, "цель 90 %", color="red", fontsize=8,
                transform=ax.get_xaxis_transform())
    axes[0].set_ylabel("MAE, метры")
    fig.suptitle("Компромисс «покрытие ↔ точность» (зелёный — финальная конфигурация)")
    fig.tight_layout()
    fig.savefig(out / "comparison_tradeoff.png", dpi=140)
    plt.close(fig)

    # --- recall / precision ---
    hr = [r for r in rows if r["recall"] is not None]
    if hr:
        fig, ax = plt.subplots(figsize=(10, 0.6 * len(hr) + 1.6))
        y = np.arange(len(hr))
        ax.barh(y - 0.18, [r["recall"] for r in hr], height=0.36,
                color="#4c72b0", label="recall-прокси")
        ax.barh(y + 0.18, [r["precision"] for r in hr], height=0.36,
                color="#dd8452", label="precision-прокси")
        ax.set_yticks(y, [r["title"] for r in hr])
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("доля (пивот внутри bbox)")
        ax.set_title("Качество детекций: recall/precision-прокси")
        ax.legend(loc="lower right")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "comparison_recall_precision.png", dpi=140)
        plt.close(fig)

    print(f"конфигураций в сравнении: {len(rows)}")
    for r in order:
        mark = " ← финал" if r["final"] else ""
        hold = f", holdout {r['holdout_mae']:.2f} м / {r['holdout_coverage'] * 100:.1f} %" \
            if r["holdout_mae"] is not None else ""
        print(f"  {r['title']:24} MAE {r['mae']:.2f} м, покрытие "
              f"{r['coverage'] * 100:.1f} %{hold}{mark}")
    print(f"графики: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
