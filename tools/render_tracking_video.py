"""Видео с наложением трекинга — в стиле веб-пульта.

Для каждой камеры пишется MP4, где на кадр наложены:

* **боксы детекций** (голубые);
* **сглаженный путь** — зелёный (измерения) / оранжевый (прогноз и достройка),
  толстая линия с ореолом, как на пульте;
* **белый пунктир** — прогноз вперёд по средней скорости;
* **серый пунктир** — пропадание данных (путь не рвётся);
* **3D-маркер** со статусом (`MEASURED` / `PREDICTED_ONLY` / `EXTRAPOLATED`);
* **GT-пивот** (жёлтый крест) — только с флагом ``--gt``;
* HUD + легенда линий.

Дополнительно пишется ``map_plan.mp4`` — панель «план X–Y (мир)» (её можно
отключить флагом ``--no-map``).

Запуск:
    py tools/render_tracking_video.py `
        --detections out/truck/det_ens_v4b_v2v1.csv `
        --estimates out/truck/est_ens_v4b_v2v1.csv `
        --out out/truck/videos

Флаги стиля: ``--trail 160 --smooth 5 --forecast-s 3 --bridge-s 3``,
панель X–Y: ``--map-trail 0`` (вся история).
"""

from __future__ import annotations

import argparse
import bisect
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np

from multiview3d.camera import project_point, project_points
from multiview3d.trail_render import (build_trail, dashed_line, draw_legend,
                                      draw_path, extrapolate, forecast,
                                      render_map_panel)
from multiview3d.truck_dataset import DEFAULT_REPAIRS, DEFAULT_ROOT, TruckDataset

COLOR_BOX = (255, 190, 0)        # BGR: голубой
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


def row_position(row: dict | None):
    """x,y,z из строки CSV; None, если координат нет."""
    if row is None:
        return None
    try:
        p = np.array([float(row["x"]), float(row["y"]), float(row["z"])])
    except (KeyError, TypeError, ValueError):
        return None
    return p if np.all(np.isfinite(p)) else None


def build_history(ests: dict[int, dict], fps: float, start: int, end: int,
                  bridge_s: float) -> list:
    """История трека по кадрам с достройкой после потери измерения."""
    hist: list = []
    for f in range(start, end + 1):
        t = f / fps
        row = ests.get(f)
        p = row_position(row)
        if p is not None:
            hist.append((t, p, str(row.get("status") or "MEASURED")))
        elif bridge_s > 0:
            q = extrapolate(hist, t, bridge_s)
            if q is not None:
                hist.append((t, q, "EXTRAPOLATED"))
    return hist


def state_at(hist: list, times: list[float], f: int, fps: float):
    """Текущая точка и её статус на кадре f (или None)."""
    t = f / fps
    k = bisect.bisect_right(times, t)
    if k and abs(times[k - 1] - t) < 1e-9:
        _, p, st = hist[k - 1]
        return np.asarray(p, dtype=float), st, k
    return None, None, k


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Видео с наложением трекинга в стиле веб-пульта")
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
    ap.add_argument("--gt", dest="draw_gt", action="store_true",
                    help="рисовать GT-пивот (по умолчанию нет, как на пульте)")
    ap.add_argument("--no-gt", dest="draw_gt", action="store_false")
    ap.set_defaults(draw_gt=False)
    ap.add_argument("--scale", type=float, default=1.0, help="уменьшить кадр (например 0.5)")
    ap.add_argument("--trail", type=int, default=160,
                    help="сколько точек пути рисовать на видео (как на пульте)")
    ap.add_argument("--map-trail", type=int, default=0,
                    help="точек на панели X–Y (0 — вся история, как на пульте)")
    ap.add_argument("--smooth", type=int, default=5,
                    help="полуокно сглаживания SG, точек (0 — выключить)")
    ap.add_argument("--forecast-s", type=float, default=3.0,
                    help="горизонт прогноза вперёд, с (0 — выключить)")
    ap.add_argument("--bridge-s", type=float, default=3.0,
                    help="достройка после пропадания, с (0 — выключить)")
    ap.add_argument("--no-map", action="store_true",
                    help="не писать map_plan.mp4")
    args = ap.parse_args(argv)

    ds = TruckDataset.discover(args.root, repairs_path=args.repairs or None)
    gt = ds.ground_truth_by_frame()
    dets = load_detections(args.detections)
    ests = load_estimates(args.estimates)

    wanted = [c.strip() for c in args.cameras.split(",")] if args.cameras else ds.names
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fps = args.fps or max(ds.fps / max(args.step, 1), 1.0)
    hist = build_history(ests, ds.fps, args.start, args.end, args.bridge_s)
    times = [h[0] for h in hist]
    print(f"история: {len(hist)} точек, видео: {args.start}..{args.end} "
          f"(шаг {args.step}), {fps:.1f} fps")

    # ---- панель плана X–Y ----
    if not args.no_map:
        map_path = out_dir / "map_plan.mp4"
        writer = cv2.VideoWriter(str(map_path), cv2.VideoWriter_fourcc(*args.codec),
                                 fps, (900, 640))
        if not writer.isOpened():
            print(f"не создать {map_path}", file=sys.stderr)
        else:
            view = None
            written = 0
            for f in range(args.start, args.end + 1, max(args.step, 1)):
                pos, status, k = state_at(hist, times, f, ds.fps)
                sub = hist[:k]
                n_map = args.map_trail if args.map_trail > 0 else len(sub)
                trail = build_trail(sub, n_map, args.smooth)
                fc = forecast(sub, args.forecast_s, args.bridge_s) if pos is not None else None
                canvas, view = render_map_panel(trail, fc, pos, status, f / ds.fps,
                                                f, view)
                writer.write(canvas)
                written += 1
            writer.release()
            size_mb = map_path.stat().st_size / 1024 ** 2 if map_path.exists() else 0.0
            print(f"map_plan: {written} кадров -> {map_path} ({size_mb:.1f} МБ)")

    # ---- камеры ----
    for cam in ds.cameras:
        if cam.name not in wanted:
            continue
        cap = cv2.VideoCapture(str(cam.video))
        if not cap.isOpened():
            print(f"{cam.name}: не открыть видео", file=sys.stderr)
            continue
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * args.scale)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * args.scale)
        ui = max(1.0, h / 540.0)      # как на пульте при 960×540, но крупнее
        styles = {
            "measured": ((20, 90, 20), max(2, int(round(8 * ui))),
                         (80, 255, 80), max(2, int(round(4 * ui)))),
            "predicted": ((0, 80, 140), max(2, int(round(6 * ui))),
                          (0, 175, 255), max(2, int(round(3 * ui)))),
        }
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

            pos, status, k = state_at(hist, times, f, ds.fps)
            sub = hist[:k]
            trail = build_trail(sub, args.trail, args.smooth)
            fc = forecast(sub, args.forecast_s, args.bridge_s) if pos is not None else None
            if calib is not None:
                if trail:
                    draw_path(frame, trail,
                              lambda P: project_points(calib, P) * args.scale,
                              styles, gap_thick=max(2, int(round(2 * ui))))
                if fc:
                    try:
                        u0, v0 = project_point(calib, fc[0])
                        u1, v1 = project_point(calib, fc[-1])
                        dashed_line(frame,
                                    (int(u0 * args.scale), int(v0 * args.scale)),
                                    (int(u1 * args.scale), int(v1 * args.scale)),
                                    (245, 245, 245), max(2, int(round(2 * ui))),
                                    dash=int(round(8 * ui)),
                                    step=int(round(16 * ui)))
                    except ValueError:
                        pass
                if pos is not None:
                    try:
                        u, v = project_point(calib, pos)
                        uu, vv = int(u * args.scale), int(v * args.scale)
                        col = (60, 200, 60) if status == "MEASURED" else (0, 175, 255)
                        cv2.drawMarker(frame, (uu, vv), col, cv2.MARKER_SQUARE,
                                       int(round(18 * ui)), max(2, int(round(2 * ui))))
                        cv2.putText(frame, f"3D {status}",
                                    (uu + int(round(12 * ui)), vv - int(round(10 * ui))),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55 * ui, col,
                                    max(1, int(round(2 * ui))), cv2.LINE_AA)
                    except ValueError:
                        pass

            if args.draw_gt and f in gt and calib is not None:
                try:
                    gu, gv = project_point(calib, gt[f])
                except ValueError:
                    gu = gv = None
                if gu is not None and 0 <= gu < calib.width and 0 <= gv < calib.height:
                    cv2.drawMarker(frame, (int(gu * args.scale), int(gv * args.scale)),
                                   COLOR_GT, cv2.MARKER_CROSS,
                                   int(round(22 * ui)), max(1, int(round(2 * ui))))

            hud = f"{cam.name}  f={f}  t={f / ds.fps:6.2f} s  det={n_det}"
            cv2.rectangle(frame, (0, 0), (int(round(620 * ui)), int(round(52 * ui))),
                          (0, 0, 0), -1)
            cv2.putText(frame, hud, (int(round(8 * ui)), int(round(21 * ui))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6 * ui, COLOR_TEXT, 1, cv2.LINE_AA)
            draw_legend(frame, ui)
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
