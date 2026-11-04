"""Таблица прогона веб-пульта: снимает `/api/stats`, пишет CSV/JSON/MD/PNG.

Запускать при работающем пульте (лучше сразу после `pecstorm launch`):

    py tools/dashboard_report.py --seconds 180

Скрипт ждёт завершения одного прохода (поле `last_run` в `/api/stats`) и
сохраняет фактические fps, покрытие и MAE против GT. Ничего не «улучшает».
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def get_stats(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(url + "/api/stats", timeout=3) as fh:
            return json.loads(fh.read().decode("utf-8"))
    except Exception:
        return None


def collect(url: str, seconds: float, interval: float) -> tuple[list[dict], dict]:
    deadline = time.time() + seconds
    samples: list[dict] = []
    last: dict = {}
    initial_run = None
    first = True
    while time.time() < deadline:
        s = get_stats(url)
        if s is not None:
            last = s
            samples.append({
                "time": datetime.now().isoformat(timespec="seconds"),
                "frame": s.get("frame"),
                "fps_now": s.get("fps_now"),
                "measured_share": s.get("measured_share"),
                "measured_frames": s.get("measured_frames"),
                "mae_m": s.get("mae_m"),
                "n_err": s.get("n_err"),
                "conf": s.get("conf"),
                "gt": s.get("gt"),
                "paused": s.get("paused"),
            })
            if first:
                initial_run = s.get("last_run")
                first = False
            run = s.get("last_run")
            if run and (initial_run is None
                        or run.get("run_id") != initial_run.get("run_id")):
                break
        time.sleep(interval)
    return samples, last


def table_png(rows: list[tuple[str, str]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 0.52 * len(rows) + 1.4))
    ax.axis("off")
    ax.set_title("Прогон веб-пульта «Пеленга»: итоговая таблица",
                 fontsize=14, fontweight="bold", pad=14)
    table = ax.table(cellText=[[k, v] for k, v in rows],
                     colLabels=["параметр", "значение"],
                     cellLoc="left", colLoc="left", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.0, 1.45)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#bbbbbb")
        if r == 0:
            cell.set_facecolor("#1b7f3b")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f2f2f2")
    fig.text(0.5, 0.02, "Источник: /api/stats во время реального прогона; "
                        "GT — только для контроля ошибки.",
             ha="center", fontsize=8.5, color="#666666")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Таблица прогона веб-пульта")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--seconds", type=float, default=180.0)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--out-base", default="results/truck/report_final/dashboard_run")
    args = ap.parse_args(argv)

    samples, last = collect(args.url, args.seconds, args.interval)
    if not samples:
        print("пульт не отвечает /api/stats", file=sys.stderr)
        return 2

    run = last.get("last_run")
    if run:
        frames = int(run["frames"])
        measured = int(run["measured_frames"])
        coverage = float(run["measured_share"])
        mae = run.get("mae_m")
        n_err = int(run.get("n_err") or 0)
        completed = True
    else:
        frames = max(int(s.get("frame") or 0) for s in samples) + 1
        measured = max(int(s.get("measured_frames") or 0) for s in samples)
        coverage = measured / max(frames, 1)
        mae = None
        n_err = 0
        completed = False

    fps = [float(s["fps_now"]) for s in samples
           if s.get("fps_now") and int(s.get("frame") or 0) > 30]
    fps_med = statistics.median(fps) if fps else None
    fps_min = min(fps) if fps else None
    fps_max = max(fps) if fps else None

    settings = {
        "model": last.get("model"),
        "imgsz": last.get("imgsz"),
        "conf": last.get("conf"),
        "max_missing": last.get("max_missing"),
        "bridge_s": last.get("bridge_s"),
    }
    summary = {
        "completed_loop": completed,
        "frames": frames,
        "measured_frames": measured,
        "measured_share": round(coverage, 4),
        "mae_m": mae,
        "n_err": n_err,
        "fps_median": round(fps_med, 2) if fps_med is not None else None,
        "fps_min": round(fps_min, 2) if fps_min is not None else None,
        "fps_max": round(fps_max, 2) if fps_max is not None else None,
        "settings": settings,
        "samples": len(samples),
        "collected_at": datetime.now().isoformat(timespec="seconds"),
    }

    base = Path(args.out_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    csv_path = base.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)
    json_path = base.with_suffix(".json")
    json_path.write_text(json.dumps({"summary": summary, "samples": samples},
                                    ensure_ascii=False, indent=2),
                         encoding="utf-8")

    rows = [
        ("Модель", str(settings["model"] or "—")),
        ("Кадров", f"{frames}"),
        ("FPS (медиана)", f"{fps_med:.2f}  (мин {fps_min:.2f}, макс {fps_max:.2f})"
         if fps else "—"),
        ("Покрытие", f"{coverage * 100:.1f} %  ({measured}/{frames})"),
        ("MAE против GT", f"{mae:.2f} м  (n={n_err})" if mae is not None else "—"),
        ("conf / imgsz", f"{settings['conf']} / {settings['imgsz']}"),
        ("max_missing / bridge", f"{settings['max_missing']} / {settings['bridge_s']} с"),
        ("Завершён полный проход", "да" if completed else "нет (по сэмплам)"),
        ("Дата", datetime.now().strftime("%d.%m.%Y %H:%M")),
    ]
    table_png(rows, base.with_suffix(".png"))

    md = ["| параметр | значение |", "|---|---|"]
    md += [f"| {k} | {v} |" for k, v in rows]
    md.append("")
    md.append("Источник: `/api/stats` во время реального прогона; GT — только "
              "для контроля ошибки.")
    base.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"полный проход: {'да' if completed else 'нет'}")
    print(f"кадров: {frames}, покрытие: {coverage * 100:.1f} %, "
          f"MAE: {mae if mae is None else round(mae, 2)} м")
    print(f"fps: медиана {fps_med if fps_med is None else round(fps_med, 2)}")
    print(f"таблица: {base.with_suffix('.png')}")
    print(f"данные:  {csv_path}, {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
