"""Таблица координат оценки vs эталона по пробегу (для dashboard_gt.mp4).

Источник: ``est_ens_v4b_v2v1.csv`` + GT датасета (те же данные, что в видео).
Пробег — накопленная длина пути GT, метры. Основной результат — CSV с BOM
(открывается в Excel без кракозябр); PNG/MD — для просмотра.

Запуск:
    py tools/make_coords_table.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from multiview3d.truck_dataset import DEFAULT_REPAIRS, DEFAULT_ROOT, TruckDataset
from render_tracking_video import load_estimates, row_position

STATUS_RU = {"MEASURED": "изм.", "PREDICTED_ONLY": "прогноз"}


def build_rows(ests: dict, gt: dict, start: int, end: int) -> list[dict]:
    rows = []
    s_gt = 0.0
    prev_gt = None
    for f in range(start, end + 1):
        g = gt.get(f)
        if g is not None:
            g = np.asarray(g, dtype=float)
            if prev_gt is not None:
                s_gt += float(np.linalg.norm(g[:2] - prev_gt[:2]))
            prev_gt = g
        row = ests.get(f)
        p = row_position(row)
        status = (row or {}).get("status", "")
        err = (float(np.linalg.norm(p - g))
               if p is not None and g is not None else None)
        d = (p - g) if p is not None and g is not None else None
        rows.append({
            "frame": f,
            "time_s": round(f / 30.0, 3),
            "status": status,
            "s_gt_m": round(s_gt, 3),
            "x_est": round(float(p[0]), 3) if p is not None else None,
            "y_est": round(float(p[1]), 3) if p is not None else None,
            "z_est": round(float(p[2]), 3) if p is not None else None,
            "x_gt": round(float(g[0]), 3) if g is not None else None,
            "y_gt": round(float(g[1]), 3) if g is not None else None,
            "z_gt": round(float(g[2]), 3) if g is not None else None,
            "dx_m": round(float(d[0]), 3) if d is not None else None,
            "dy_m": round(float(d[1]), 3) if d is not None else None,
            "dz_m": round(float(d[2]), 3) if d is not None else None,
            "error_m": round(err, 3) if err is not None else None,
        })
    return rows


def sample_by_distance(rows: list[dict], step_m: float) -> list[dict]:
    out = []
    next_s = 0.0
    for r in rows:
        if r["s_gt_m"] + 1e-9 >= next_s:
            out.append(r)
            next_s = r["s_gt_m"] + step_m
    if rows and out[-1]["frame"] != rows[-1]["frame"]:
        out.append(rows[-1])
    return out


def fmt(v, nd=2):
    return "—" if v is None else f"{v:.{nd}f}"


def markdown(rows: list[dict], path: Path, step_m: float) -> None:
    lines = [
        f"| пробег, м | кадр | t, с | статус | X оц | Y оц | Z оц | "
        f"X эт | Y эт | Z эт | ошибка, м |",
        "|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['s_gt_m']:.1f} | {r['frame']} | {r['time_s']:.2f} | "
            f"{STATUS_RU.get(r['status'], r['status'] or '—')} | "
            f"{fmt(r['x_est'])} | {fmt(r['y_est'])} | {fmt(r['z_est'])} | "
            f"{fmt(r['x_gt'])} | {fmt(r['y_gt'])} | {fmt(r['z_gt'])} | "
            f"{fmt(r['error_m'])} |")
    lines += ["", f"Шаг таблицы: {step_m:g} м пробега; полный CSV — рядом. "
                  "Источник: `est_ens_v4b_v2v1.csv` + GT датасета."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def png_table(rows: list[dict], path: Path, step_m: float) -> None:
    head = ["пробег", "кадр", "t, с", "статус", "X оц", "Y оц", "Z оц",
            "X эт", "Y эт", "Z эт", "err"]
    data = []
    for r in rows:
        data.append([
            f"{r['s_gt_m']:.1f}", str(r["frame"]), f"{r['time_s']:.2f}",
            STATUS_RU.get(r["status"], r["status"] or "—"),
            fmt(r["x_est"]), fmt(r["y_est"]), fmt(r["z_est"]),
            fmt(r["x_gt"]), fmt(r["y_gt"]), fmt(r["z_gt"]),
            fmt(r["error_m"]),
        ])
    half = (len(data) + 1) // 2
    parts = [data[:half], data[half:]]
    fig, axes = plt.subplots(1, 2, figsize=(17, max(6.0, 0.26 * half + 2.2)))
    fig.suptitle("Координаты оценки и эталона по пробегу — dashboard_gt.mp4",
                 fontsize=15, fontweight="bold")
    for ax, part in zip(axes, parts):
        ax.axis("off")
        if not part:
            continue
        table = ax.table(cellText=part, colLabels=head, cellLoc="center",
                         colLoc="center", loc="upper center")
        table.auto_set_font_size(False)
        table.set_fontsize(7.0)
        table.scale(1.0, 1.32)
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor("#cccccc")
            if r == 0:
                cell.set_facecolor("#1b7f3b")
                cell.set_text_props(color="white", fontweight="bold")
            elif r % 2 == 0:
                cell.set_facecolor("#f3f3f3")
    fig.text(0.5, 0.01, f"Шаг: {step_m:g} м пробега. Источник: "
                        "est_ens_v4b_v2v1.csv + GT; пробег — длина GT-пути.",
             ha="center", fontsize=8.5, color="#666666")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


CSV_HEADERS = ["Пробег, м", "Кадр", "Время, с", "Статус",
               "X оц, м", "Y оц, м", "Z оц, м",
               "X эт, м", "Y эт, м", "Z эт, м",
               "dX, м", "dY, м", "dZ, м", "Ошибка, м"]
CSV_KEYS = ["s_gt_m", "frame", "time_s", "status", "x_est", "y_est", "z_est",
            "x_gt", "y_gt", "z_gt", "dx_m", "dy_m", "dz_m", "error_m"]


def write_csv(path: Path, data: list[dict]) -> None:
    """CSV с BOM и русскими заголовками — открывается в Excel без кракозябр."""
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_HEADERS)
        for r in data:
            values = []
            for k in CSV_KEYS:
                v = r.get(k)
                if k == "status":
                    v = STATUS_RU.get(v, v or "—")
                values.append("" if v is None else v)
            writer.writerow(values)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Таблица координат по пробегу")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--repairs", default=str(DEFAULT_REPAIRS))
    ap.add_argument("--estimates", default="out/truck/est_ens_v4b_v2v1.csv")
    ap.add_argument("--out-base", default="results/truck/report_final/dashboard_coords")
    ap.add_argument("--step-m", type=float, default=10.0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=1000)
    args = ap.parse_args(argv)

    ds = TruckDataset.discover(args.root, repairs_path=args.repairs or None)
    gt = ds.ground_truth_by_frame()
    ests = load_estimates(args.estimates)
    rows = build_rows(ests, gt, args.start, args.end)
    sampled = sample_by_distance(rows, args.step_m)

    base = Path(args.out_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    write_csv(base.with_suffix(".csv"), rows)
    write_csv(base.with_name(base.stem + "_sampled.csv"), sampled)
    markdown(sampled, base.with_suffix(".md"), args.step_m)
    png_table(sampled, base.with_suffix(".png"), args.step_m)

    total = rows[-1]["s_gt_m"] if rows else 0.0
    errs = [r["error_m"] for r in rows if r["error_m"] is not None]
    print(f"пробег GT: {total:.1f} м; строк всего: {len(rows)}, "
          f"в таблице: {len(sampled)} (шаг {args.step_m:g} м)")
    if errs:
        print(f"MAE по строкам с оценкой: {np.mean(errs):.3f} м (n={len(errs)})")
    print(f"csv (все кадры): {base.with_suffix('.csv')}")
    print(f"csv (по пробегу): {base.with_name(base.stem + '_sampled.csv')}")
    print(f"таблица: {base.with_suffix('.png')}")
    print(f"markdown: {base.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
