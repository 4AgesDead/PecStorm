"""Полный цикл оценки дообученной модели: детекция → якоря → 3D → метрики.

Один вызов прогоняет все стадии для указанного чекпойнта и печатает сводку
(MAE, покрытие, recall-прокси) плюс сравнение с базовой моделью, если задан
``--baseline-eval``.

Запуск (в окружении с torch; для GPU — из ``.venv-rocm``):
    python tools/run_finetuned_eval.py --model runs/.../weights/best.pt --device 0 --tag v2
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def run(cmd: list[str]) -> None:
    print("→ " + " ".join(cmd))
    res = subprocess.run(cmd, text=True, encoding="utf-8", errors="replace",
                         capture_output=True)
    if res.returncode != 0:
        print(res.stdout[-2000:])
        print(res.stderr[-2000:], file=sys.stderr)
        raise SystemExit(f"команда упала: {cmd[1]}")
    tail = [ln for ln in res.stdout.strip().splitlines() if ln.strip()][-3:]
    for ln in tail:
        print("   " + ln)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Оценка дообученной модели целиком")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--max-reprojection-px", type=float, default=50.0)
    ap.add_argument("--fit-end", type=int, default=700)
    ap.add_argument("--holdout-start", type=int, default=700)
    ap.add_argument("--out-dir", default="out/truck")
    ap.add_argument("--baseline-eval", default=None,
                    help="JSON базовой оценки для сравнения (например out/truck/evaluation.json)")
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    det_csv = out / f"detections_{args.tag}.csv"
    det_stats = out / f"detect_stats_{args.tag}.json"
    anchors = out / f"anchor_{args.tag}.json"
    est_csv = out / f"est_{args.tag}.csv"
    est_json = out / f"est_{args.tag}.json"
    eval_json = out / f"eval_{args.tag}.json"
    eval_csv = out / f"evalpf_{args.tag}.csv"
    eval_hold = out / f"eval_{args.tag}_holdout.json"
    hitrate = out / f"hitrate_{args.tag}.json"
    py = sys.executable

    run([py, "apps/detect_truck.py", "--start", "0", "--end", "1000", "--step", "1",
         "--model", args.model, "--device", args.device, "--conf", str(args.conf),
         "--imgsz", str(args.imgsz), "--class-names", "truck",
         "--out", str(det_csv), "--stats", str(det_stats)])
    run([py, "tools/anchor_fit_views.py", "--detections", str(det_csv),
         "--max-reprojection-px", str(args.max_reprojection_px),
         "--fit-start", "0", "--fit-end", str(args.fit_end),
         "--test-start", str(args.holdout_start), "--test-end", "1000",
         "--out", str(anchors)])
    azimuth = out / f"azimuth_{args.tag}.json"
    run([py, "tools/anchor_fit_azimuth.py", "--detections", str(det_csv),
         "--anchor-views", str(anchors),
         "--max-reprojection-px", str(args.max_reprojection_px),
         "--fit-end", "1000", "--out", str(azimuth)])
    run([py, "apps/reconstruct_truck.py", "--detections", str(det_csv),
         "--max-reprojection-px", str(args.max_reprojection_px),
         "--anchor-offset-views", str(anchors),
         "--anchor-azimuth", str(azimuth),
         "--out-csv", str(est_csv), "--out-json", str(est_json)])
    run([py, "apps/evaluate_truck.py", "--estimates", str(est_csv),
         "--out-json", str(eval_json), "--out-csv", str(eval_csv)])
    run([py, "apps/evaluate_truck.py", "--estimates", str(est_csv),
         "--start", str(args.holdout_start), "--end", "1000",
         "--out-json", str(eval_hold), "--out-csv", str(out / f"evalpf_{args.tag}_holdout.csv")])
    run([py, "experiments/anchor_hitrate.py", "--detections", str(det_csv),
         "--out", str(hitrate)])

    def load(p):
        return json.loads(Path(p).read_text(encoding="utf-8"))

    cur, hold, hit = load(eval_json), load(eval_hold), load(hitrate)
    summary = {
        "tag": args.tag,
        "model": args.model,
        "measured": cur["measured"],
        "coverage": cur["coverage"],
        "holdout": hold["measured"],
        "holdout_coverage": hold["coverage"],
        "hitrate": hit["totals"],
    }
    (out / f"summary_{args.tag}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    def fmt(d, c):
        m = d["measured"]
        return (f"n={m['n_points']}, MAE={m['mae_mm'] / 1000:.2f} м, "
                f"медиана={m['median_mm'] / 1000:.2f}, P95={m['p95_mm'] / 1000:.2f}, "
                f"покрытие={c['measured']:.3f}")

    print(f"\n=== {args.tag} ===")
    print("все кадры   : " + fmt(cur, cur["coverage"]))
    print("holdout 700+ : " + fmt(hold, hold["coverage"]))
    t = hit["totals"]
    print(f"recall_proxy={t['recall_proxy']}, precision_proxy={t['precision_proxy']}, "
          f"медиана центр→пивот={t['dist_center_to_pivot_px_median']} px")
    if args.baseline_eval:
        base = load(args.baseline_eval)
        bm = base["measured"]
        print(f"база: MAE={bm['mae_mm'] / 1000:.2f} м, покрытие={base['coverage']['measured']:.3f}"
              f"  →  улучшение MAE {bm['mae_mm'] / max(cur['measured']['mae_mm'], 1e-9):.2f}×")
    print(f"сводка: {out / f'summary_{args.tag}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
