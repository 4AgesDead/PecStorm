"""Видео «как веб-пульт» для scene2: 9 камер + план X–Y + таблица.

Без GT: в таблице нет MAE, только фактические MEASURED/INTERPOLATED и доля
кадров с реальным измерением. Это честная визуализация submission_scene2.csv.

Запуск:
    py tools/render_scene2_video.py `
        --submission results/scene2/submission_scene2.csv `
        --out out/scene2/scene2_dashboard.mp4
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

from multiview3d.camera import project_point, project_points
from multiview3d.scene2_dataset import Scene2Dataset
from multiview3d.trail_render import (STYLES_VIDEO, draw_path, render_map_panel)

W, H = 1920, 1080
TOP = 60
TABLE_Y = 870
TILE_W, TILE_H = 480, 270


def load_submission(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "frame": int(r["frame"]),
                "t": float(r.get("timestamp") or int(r["frame"]) / 30.0),
                "p": np.array([float(r["x"]), float(r["y"]), float(r["z"])]),
                "status": r.get("status") or "INTERPOLATED",
                "views": r.get("views") or "",
            })
    rows.sort(key=lambda x: x["frame"])
    return rows


def border(canvas, x, y, title):
    cv2.rectangle(canvas, (x, y), (x + TILE_W - 1, y + TILE_H - 1),
                  (60, 60, 60), 2)
    cv2.rectangle(canvas, (x, y), (x + TILE_W, y + 24), (0, 0, 0), -1)
    cv2.putText(canvas, title, (x + 7, y + 17), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (235, 235, 235), 1, cv2.LINE_AA)


def camera_tile(frame, calib, trail, pos, status, name, f, fps, scale):
    img = cv2.resize(frame, (int(frame.shape[1] * scale),
                             int(frame.shape[0] * scale)),
                     interpolation=cv2.INTER_AREA)
    if trail:
        draw_path(img, trail, lambda P: project_points(calib, P) * scale,
                  STYLES_VIDEO, gap_thick=2)
    if pos is not None:
        try:
            u, v = project_point(calib, pos)
            uu, vv = int(u * scale), int(v * scale)
            col = (60, 200, 60) if status == "MEASURED" else (0, 175, 255)
            cv2.drawMarker(img, (uu, vv), col, cv2.MARKER_SQUARE, 16, 2)
            cv2.putText(img, status, (uu + 9, vv - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
        except ValueError:
            pass
    cv2.rectangle(img, (0, 0), (int(300 * scale), int(24 * scale)),
                  (0, 0, 0), -1)
    cv2.putText(img, f"{name} f={f} t={f / fps:5.1f}s", (5, int(17 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42 * scale, (240, 240, 240), 1,
                cv2.LINE_AA)
    return cv2.resize(img, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)


def info_panel(title, lines, x, y):
    canvas = np.full((TILE_H, TILE_W, 3), (12, 14, 18), np.uint8)
    cv2.putText(canvas, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (235, 235, 235), 2, cv2.LINE_AA)
    for i, (text, color) in enumerate(lines):
        cv2.putText(canvas, text, (10, 64 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
    return canvas


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Видео scene2 «как пульт»")
    ap.add_argument("--root", default="scene2")
    ap.add_argument("--submission", default="results/scene2/submission_scene2.csv")
    ap.add_argument("--out", default="out/scene2/scene2_dashboard.mp4")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=1931)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--cam-scale", type=float, default=0.5)
    ap.add_argument("--trail", type=int, default=160)
    ap.add_argument("--codec", default="mp4v")
    args = ap.parse_args(argv)

    ds = Scene2Dataset.discover(args.root)
    rows = load_submission(Path(args.submission))
    by_frame = {r["frame"]: r for r in rows}
    caps = {c.name: cv2.VideoCapture(str(c.video)) for c in ds.cameras}
    for cap in caps.values():
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*args.codec),
                             args.fps, (W, H))
    if not writer.isOpened():
        print(f"не создать {out}", file=sys.stderr)
        return 2

    measured = 0
    map_view = None
    for f in range(args.start, args.end + 1):
        frames = {}
        for name, cap in caps.items():
            ok, img = cap.read()
            frames[name] = img if ok else np.zeros((1080, 1920, 3), np.uint8)
        row = by_frame.get(f)
        pos = row["p"] if row else None
        status = row["status"] if row else "—"
        if status == "MEASURED":
            measured += 1
        processed = f - args.start + 1
        share = measured / max(processed, 1)

        canvas = np.full((H, W, 3), (18, 20, 24), np.uint8)
        cv2.rectangle(canvas, (0, 0), (W, TOP), (0, 0, 0), -1)
        cv2.putText(canvas, "Пеленга 2 · scene2 · 9 камер + план X–Y "
                            "(COCO yolov8x, без GT)",
                    (30, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (235, 235, 235), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"f={f}  t={f / args.fps:.2f}s  "
                            f"измерено {measured}/{processed} ({share * 100:.1f}%)",
                    (1350, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (200, 200, 200), 1, cv2.LINE_AA)

        trail = []
        for r in rows:
            if r["frame"] > f:
                break
            trail.append((r["p"], r["status"]))
        all_trail = trail
        trail = all_trail[-args.trail:] if args.trail > 0 else all_trail
        map_trail = all_trail[-600:]

        # карта
        map_c, map_view = render_map_panel(
            map_trail, None, pos, status, f / args.fps, f, map_view,
            size=(800, 450))
        map_tile = cv2.resize(map_c, (TILE_W, TILE_H),
                              interpolation=cv2.INTER_AREA)
        canvas[TOP:TOP + TILE_H, 0:TILE_W] = map_tile
        border(canvas, 0, TOP, "план X–Y (мир)")

        # 9 камер: слоты 1..9
        for ci, cam in enumerate(ds.cameras):
            slot = ci + 1
            x = (slot % 4) * TILE_W
            y = TOP + (slot // 4) * TILE_H
            calib = ds.provider.calibrations_at(f, nearest=True).get(cam.name)
            tile = camera_tile(frames[cam.name], calib, trail, pos, status,
                               cam.name, f, args.fps, args.cam_scale)
            canvas[y:y + TILE_H, x:x + TILE_W] = tile
            border(canvas, x, y, cam.name)

        # свободные слоты: метрики и легенда
        mx, my = 2 * TILE_W, TOP + 2 * TILE_H
        info = info_panel("", [
            (f"кадров: {processed} / {args.end - args.start + 1}", (220, 220, 220)),
            (f"MEASURED: {measured}", (80, 255, 80)),
            (f"INTERPOLATED: {processed - measured}", (0, 175, 255)),
            (f"доля измерений: {share * 100:.1f}%", (220, 220, 220)),
            ("GT: нет — MAE не считается", (0, 220, 255)),
        ], mx, my)
        canvas[my:my + TILE_H, mx:mx + TILE_W] = info
        border(canvas, mx, my, "статистика")

        lx = 3 * TILE_W
        legend = info_panel("", [
            ("—— измерение (MEASURED)", (80, 255, 80)),
            ("—— интерполяция", (0, 175, 255)),
            ("□ текущая точка", (220, 220, 220)),
            ("жёлтый GT не рисуется: GT нет", (160, 160, 160)),
        ], lx, my)
        canvas[my:my + TILE_H, lx:lx + TILE_W] = legend
        border(canvas, lx, my, "легенда")

        # нижняя таблица
        cv2.rectangle(canvas, (0, TABLE_Y), (W, H), (0, 0, 0), -1)
        cv2.putText(canvas, "scene2 · 3D-координаты грузовика по 9 камерам",
                    (40, TABLE_Y + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    (235, 235, 235), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"кадр {f} / {args.end}   t={f / args.fps:6.2f} с   "
                            f"статус: {status}",
                    (40, TABLE_Y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"измерено: {measured}   интерполяция: "
                            f"{processed - measured}   доля измерений: {share * 100:.1f}%",
                    (40, TABLE_Y + 122), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (80, 255, 80), 2, cv2.LINE_AA)
        cv2.putText(canvas, "GT в scene2 нет: MAE/покрытие против эталона не считаются; "
                            "интерполяция — не измерение.",
                    (40, TABLE_Y + 164), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "источник: submission_scene2.csv · "
                            "detector COCO yolov8x · 9 камер",
                    (40, TABLE_Y + 198), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (150, 150, 150), 1, cv2.LINE_AA)

        writer.write(canvas)
        if f % 100 == 0:
            print(f"  кадр {f}/{args.end}")

    for cap in caps.values():
        cap.release()
    writer.release()
    size_mb = out.stat().st_size / 1024 ** 2 if out.exists() else 0.0
    print(f"готово: {out} ({size_mb:.1f} МБ)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
