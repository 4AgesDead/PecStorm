"""Видео с наложением трекинга: боксы детекций + проекция 3D-оценки + GT.

Для каждой камеры пишется MP4, где на кадр наложены:

* **боксы детекций** (голубые) — то, что нашёл детектор;
* **проекция 3D-оценки** (зелёный квадрат/крест) — куда конвейер поставил
  грузовик; подпись со статусом (`MEASURED`/`PREDICTED_ONLY`) и числом видов;
* **GT-пивот** (жёлтый крест) — эталон, если он есть в кадре;
* HUD: кадр, время, число детекций, покрытие.

Запуск:
    python tools/render_tracking_video.py --detections out/truck/detections_ens.csv `
        --estimates out/truck/est_ens.csv --out out/truck/videos
    python tools/render_tracking_video.py ... --step 2 --fps 15 --cameras cam3,cam5
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np

from multiview3d.camera import project_point
from multiview3d.truck_dataset import DEFAULT_REPAIRS, DEFAULT_ROOT, TruckDataset

COLOR_BOX = (255, 190, 0)        # BGR: голубой
COLOR_EST = (60, 200, 60)        # зелёный
COLOR_GT = (0, 220, 255)         # жёлтый
COLOR_TEXT = (255, 255, 255)


def load_detections(path: str | Path) -> dict[tuple[str, int], list[tuple[float, float, float, float, float]]]:
    out: dict[tuple[str, int], list] = {}
    with Path(path).open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("x1") in (None, ""):
                continue
            key = (r["camera"], int(r["frame"]))
            out.setdefault(key, []).append(
                (float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"]),
                 float(r["score"])))
    return out


def load_estimates(path: str | Path) -> dict[int, dict]:
    """Лучшая оценка на кадр: MEASURED приоритетнее."""
    best: dict[int, dict] = {}
    with Path(path).open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            f = int(r["frame"])
            rank = 2 if r["status"] == "MEASURED" else (
                1 if r["status"] == "PREDICTED_ONLY" else 0)
            cur = best.get(f)
            if cur is None or rank > cur["rank"]:
                best[f] = {"rank": rank, "row": r}
    return {f: v["row"] for f, v in best.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Видео с наложением трекинга")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--repairs", default=str(DEFAULT_REPAIRS))
    ap.add_argument("--detections", required=True)
    ap.add_argument("--estimates", required=True)
    ap.add_argument("--out", default="out/truck/videos")
    ap.add_argument("--cameras", default=None, help="например cam3,cam5 (по умолчанию все)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=1000)
    ap.add_argument("--step", type=int, default=1,
                    help="брать каждый N-й кадр (2 — вдвое короче видео)")
    ap.add_argument("--fps", type=float, default=None,
                    help="FPS выходного видео (по умолчанию fps/step)")
    ap.add_argument("--codec", default="mp4v")
    ap.add_argument("--draw-gt", action="store_true", default=True)
    ap.add_argument("--no-gt", dest="draw_gt", action="store_false")
    ap.add_argument("--scale", type=float, default=1.0, help="уменьшить кадр (например 0.5)")
    args = ap.parse_args(argv)

    ds = TruckDataset.discover(args.root, repairs_path=args.repairs or None)
    gt = ds.ground_truth_by_frame()
    dets = load_detections(args.detections)
    ests = load_estimates(args.estimates)

    wanted = [c.strip() for c in args.cameras.split(",")] if args.cameras else ds.names
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fps = args.fps or max(ds.fps / max(args.step, 1), 1.0)

    for cam in ds.cameras:
        if cam.name not in wanted:
            continue
        cap = cv2.VideoCapture(str(cam.video))
        if not cap.isOpened():
            print(f"{cam.name}: не открыть видео", file=sys.stderr)
            continue
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * args.scale)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * args.scale)
        path = out_dir / f"{cam.name}_tracking.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*args.codec),
                                 fps, (w, h))
        if not writer.isOpened():
            print(f"{cam.name}: не создать {path}", file=sys.stderr)
            cap.release()
            continue

        f = 0
        written = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if f < args.start or f > args.end or (f - args.start) % args.step:
                f += 1
                continue
            if args.scale != 1.0:
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            calib = ds.provider.calibrations_at(f, nearest=True).get(cam.name)
            n_det = 0
            for (x1, y1, x2, y2, score) in dets.get((cam.name, f), []):
                p1 = (int(x1 * args.scale), int(y1 * args.scale))
                p2 = (int(x2 * args.scale), int(y2 * args.scale))
                cv2.rectangle(frame, p1, p2, COLOR_BOX, 2)
                cv2.putText(frame, f"{score:.2f}", (p1[0], max(p1[1] - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BOX, 1, cv2.LINE_AA)
                n_det += 1

            if calib is not None and f in ests:
                row = ests[f]
                P = np.array([float(row["x"]), float(row["y"]), float(row["z"])])
                try:
                    u, v = project_point(calib, P)
                except ValueError:
                    u = v = None
                if u is not None and np.isfinite(u) and np.isfinite(v):
                    uu, vv = int(u * args.scale), int(v * args.scale)
                    s = 9
                    cv2.rectangle(frame, (uu - s, vv - s), (uu + s, vv + s), COLOR_EST, 2)
                    cv2.drawMarker(frame, (uu, vv), COLOR_EST, cv2.MARKER_CROSS, 18, 1)
                    label = f"3D {row['status']} ({row.get('views') or '-'})"
                    cv2.putText(frame, label, (uu + 14, vv - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_EST, 2, cv2.LINE_AA)
            if args.draw_gt and f in gt and calib is not None:
                try:
                    gu, gv = project_point(calib, gt[f])
                except ValueError:
                    gu = gv = None
                if gu is not None and 0 <= gu < calib.width and 0 <= gv < calib.height:
                    cv2.drawMarker(frame, (int(gu * args.scale), int(gv * args.scale)),
                                   COLOR_GT, cv2.MARKER_CROSS, 22, 2)

            hud = f"{cam.name}  f={f}  t={f / ds.fps:6.2f} s  det={n_det}"
            cv2.rectangle(frame, (0, 0), (int(430 * args.scale), int(28 * args.scale)),
                          (0, 0, 0), -1)
            cv2.putText(frame, hud, (8, int(20 * args.scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1, cv2.LINE_AA)
            writer.write(frame)
            written += 1
            f += 1
        cap.release()
        writer.release()
        size_mb = path.stat().st_size / 1024 ** 2 if path.exists() else 0.0
        print(f"{cam.name}: {written} кадров -> {path} ({size_mb:.1f} МБ)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
