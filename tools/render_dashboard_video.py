"""Запись «как веб-пульт»: 5 камер + план X–Y + таблица метрик.

Это честная запись: покрытие и MAE считаются по реальным оценкам и GT.
Целевые отметки в таблицу не добавляются — только фактические числа.

Запуск:
    py tools/render_dashboard_video.py `
        --detections out/truck/det_ens_v4b_v2v1.csv `
        --estimates out/truck/est_ens_v4b_v2v1.csv `
        --out out/truck/videos/dashboard_gt.mp4 --gt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np

from multiview3d.camera import project_point, project_points
from multiview3d.trail_render import (STYLES_VIDEO, build_trail, dashed_line,
                                      draw_path, forecast, render_map_panel)
from multiview3d.truck_dataset import DEFAULT_REPAIRS, DEFAULT_ROOT, TruckDataset
from render_tracking_video import (build_history, load_detections,
                                   load_estimates, row_position, state_at)

W, H = 1920, 1080
TILE_W, TILE_H = 640, 360
TOP = 60
BOTTOM = 780


def draw_tile_border(canvas, x, y, title):
    cv2.rectangle(canvas, (x, y), (x + TILE_W - 1, y + TILE_H - 1),
                  (60, 60, 60), 2)
    cv2.rectangle(canvas, (x, y), (x + TILE_W, y + 26), (0, 0, 0), -1)
    cv2.putText(canvas, title, (x + 8, y + 19), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (235, 235, 235), 1, cv2.LINE_AA)


def render_camera(frame, calib, dets, trail, fc, pos, status, gt_pos,
                  name, f, fps, scale):
    img = cv2.resize(frame, (int(frame.shape[1] * scale),
                             int(frame.shape[0] * scale)),
                     interpolation=cv2.INTER_AREA)
    for (x1, y1, x2, y2, score) in dets:
        p1 = (int(x1 * scale), int(y1 * scale))
        p2 = (int(x2 * scale), int(y2 * scale))
        cv2.rectangle(img, p1, p2, (255, 190, 0), 2)
        cv2.putText(img, f"{score:.2f}", (p1[0], max(p1[1] - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 190, 0), 1, cv2.LINE_AA)
    if calib is not None and trail:
        draw_path(img, trail, lambda P: project_points(calib, P) * scale,
                  STYLES_VIDEO, gap_thick=2)
        if fc:
            try:
                u0, v0 = project_point(calib, fc[0])
                u1, v1 = project_point(calib, fc[-1])
                dashed_line(img, (int(u0 * scale), int(v0 * scale)),
                            (int(u1 * scale), int(v1 * scale)),
                            (245, 245, 245), 2, dash=8, step=16)
            except ValueError:
                pass
        if pos is not None:
            try:
                u, v = project_point(calib, pos)
                uu, vv = int(u * scale), int(v * scale)
                col = (60, 200, 60) if status == "MEASURED" else (0, 175, 255)
                cv2.drawMarker(img, (uu, vv), col, cv2.MARKER_SQUARE, 16, 2)
                cv2.putText(img, f"3D {status}", (uu + 10, vv - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)
            except ValueError:
                pass
        if gt_pos is not None:
            try:
                gu, gv = project_point(calib, gt_pos)
                cv2.drawMarker(img, (int(gu * scale), int(gv * scale)),
                               (0, 220, 255), cv2.MARKER_CROSS, 18, 2)
            except ValueError:
                pass
    hud = f"{name}  f={f}  t={f / fps:6.2f} s"
    cv2.rectangle(img, (0, 0), (int(360 * scale), int(26 * scale)),
                  (0, 0, 0), -1)
    cv2.putText(img, hud, (6, int(18 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                0.5 * scale, (240, 240, 240), 1, cv2.LINE_AA)
    return cv2.resize(img, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)


def draw_table(canvas, f, end, fps, status, coverage, mae, n_err,
               label=""):
    cv2.rectangle(canvas, (0, BOTTOM), (W, H), (0, 0, 0), -1)
    cv2.line(canvas, (0, BOTTOM), (W, BOTTOM), (70, 70, 70), 2)
    title = "Пеленга · запись веб-пульта: 5 камер + план X–Y"
    if label:
        title += f"  ·  {label}"
    cv2.putText(canvas, title, (40, BOTTOM + 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (235, 235, 235), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"кадр {f} / {end}    t = {f / fps:6.2f} с    "
                        f"статус: {status or '—'}",
                (40, BOTTOM + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (200, 200, 200), 1, cv2.LINE_AA)

    # покрытие: только фактическое значение
    cv2.putText(canvas, f"покрытие: {coverage * 100:5.1f} %", (40, BOTTOM + 126),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (80, 255, 80), 2, cv2.LINE_AA)
    bx, by, bw, bh = 40, BOTTOM + 140, 760, 22
    cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (70, 70, 70), 1)
    fill = int(round(bw * min(max(coverage, 0.0), 1.0)))
    cv2.rectangle(canvas, (bx, by), (bx + fill, by + bh), (70, 190, 70), -1)

    cv2.putText(canvas, f"MAE: {mae:5.2f} м   (MEASURED, n={n_err})",
                (40, BOTTOM + 205), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (235, 235, 235), 2, cv2.LINE_AA)
    cv2.putText(canvas, "источник оценок: ансамбль v1+v2+v4b + якоря; "
                        "GT — только для контроля, в оценку не входит",
                (40, BOTTOM + 245), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (160, 160, 160), 1, cv2.LINE_AA)

    # легенда
    lx, ly = 960, BOTTOM + 80
    items = [((80, 255, 80), "измерено"), ((0, 175, 255), "прогноз"),
             ((245, 245, 245), "вперёд"), ((145, 145, 145), "разрыв"),
             ((0, 220, 255), "эталон GT")]
    for i, (col, label) in enumerate(items):
        y = ly + i * 30
        cv2.line(canvas, (lx, y), (lx + 34, y), col, 3)
        cv2.putText(canvas, label, (lx + 46, y + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1,
                    cv2.LINE_AA)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Видео «как веб-пульт»")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--repairs", default=str(DEFAULT_REPAIRS))
    ap.add_argument("--detections", default="out/truck/det_ens_v4b_v2v1.csv")
    ap.add_argument("--estimates", default="out/truck/est_ens_v4b_v2v1.csv")
    ap.add_argument("--out", default="out/truck/videos/dashboard.mp4")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=1000)
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--codec", default="mp4v")
    ap.add_argument("--cam-scale", type=float, default=0.5)
    ap.add_argument("--trail", type=int, default=160)
    ap.add_argument("--map-trail", type=int, default=0)
    ap.add_argument("--smooth", type=int, default=5)
    ap.add_argument("--forecast-s", type=float, default=3.0)
    ap.add_argument("--bridge-s", type=float, default=3.0)
    ap.add_argument("--label", default="",
                    help="подпись отрезка, например 'фрагмент 0..213'")
    ap.add_argument("--gt", action="store_true",
                    help="показывать эталон GT на камерах и карте")
    ap.add_argument("--no-map", action="store_true")
    args = ap.parse_args(argv)

    ds = TruckDataset.discover(args.root, repairs_path=args.repairs or None)
    gt = ds.ground_truth_by_frame()
    dets = load_detections(args.detections)
    ests = load_estimates(args.estimates)
    fps = float(args.fps or ds.fps)
    hist = build_history(ests, ds.fps, args.start, args.end, args.bridge_s)
    times = [h[0] for h in hist]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*args.codec),
                             fps, (W, H))
    if not writer.isOpened():
        print(f"не создать {out}", file=sys.stderr)
        return 2

    caps = [cv2.VideoCapture(str(c.video)) for c in ds.cameras]
    map_view = None
    for f in range(args.start, args.end + 1):
        frames = []
        for cap in caps:
            ok, img = cap.read()
            frames.append(img if ok else np.zeros((1080, 1920, 3), np.uint8))

        pos, status, k = state_at(hist, times, f, ds.fps)
        sub = hist[:k]
        trail = build_trail(sub, args.trail, args.smooth)
        n_map = args.map_trail if args.map_trail > 0 else len(sub)
        map_trail = build_trail(sub, n_map, args.smooth)
        fc = forecast(sub, args.forecast_s, args.bridge_s) if pos is not None else None

        # накопленные метрики по факту
        measured_total = sum(1 for ff in range(args.start, f + 1)
                             if (ests.get(ff) or {}).get("status") == "MEASURED")
        processed = f - args.start + 1
        coverage = measured_total / max(processed, 1)
        err_sum, n_err = 0.0, 0
        for ff in range(args.start, f + 1):
            rr = ests.get(ff)
            if (rr or {}).get("status") != "MEASURED" or ff not in gt:
                continue
            pp = row_position(rr)
            if pp is not None:
                err_sum += float(np.linalg.norm(pp - np.asarray(gt[ff])))
                n_err += 1
        mae = err_sum / n_err if n_err else 0.0

        canvas = np.full((H, W, 3), (18, 20, 24), np.uint8)
        cv2.rectangle(canvas, (0, 0), (W, TOP), (0, 0, 0), -1)
        cv2.putText(canvas, "Пеленга · трекинг в реальном времени "
                            "(запись пульта, ансамбль v1+v2+v4b)",
                    (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (235, 235, 235), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"f={f}  t={f / fps:.2f} s  "
                            f"покрытие {coverage * 100:.1f}%  MAE {mae:.2f} м",
                    (1300, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (200, 200, 200), 1, cv2.LINE_AA)

        # карта
        if not args.no_map:
            map_canvas, map_view = render_map_panel(
                map_trail, fc, pos, status, f / fps, f,
                map_view, size=(800, 450),
                gt_points=([gt[k] for k in sorted(gt)] if args.gt else None),
                gt_position=(gt.get(f) if args.gt else None))
            map_tile = cv2.resize(map_canvas, (TILE_W, TILE_H),
                                  interpolation=cv2.INTER_AREA)
            canvas[TOP:TOP + TILE_H, 0:TILE_W] = map_tile
            draw_tile_border(canvas, 0, TOP, "план X–Y (мир)")

        # камеры: слоты 1..5 при карте, иначе 0..4
        for ci, cam in enumerate(ds.cameras):
            slot = ci if args.no_map else ci + 1
            x = (slot % 3) * TILE_W
            y = TOP + (slot // 3) * TILE_H
            calib = ds.provider.calibrations_at(f, nearest=True).get(cam.name)
            tile = render_camera(
                frames[ci], calib, dets.get((cam.name, f), []), trail, fc,
                pos, status, gt.get(f) if args.gt else None,
                cam.name, f, fps, args.cam_scale)
            canvas[y:y + TILE_H, x:x + TILE_W] = tile
            draw_tile_border(canvas, x, y, cam.name)

        draw_table(canvas, f, args.end, fps, status, coverage, mae, n_err,
                   args.label)
        writer.write(canvas)
        if f % 100 == 0:
            print(f"  кадр {f}/{args.end}")

    for cap in caps:
        cap.release()
    writer.release()
    size_mb = out.stat().st_size / 1024 ** 2 if out.exists() else 0.0
    print(f"готово: {out} ({size_mb:.1f} МБ)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
