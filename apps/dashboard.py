"""Веб-приложение «пульт»: живые камеры с трекингом и управлением.

Что делает:
* читает 5 MP4 «Пеленги» в темпе, близком к реальному (или быстрее — как успевает);
* на каждом кадре запускает детектор, слияние и трекинг существующим конвейером
  (`MultiViewPipeline.step`), применяет якоря по наборам видов и азимутальную
  поправку к последней точке трека;
* отдаёт в браузер MJPEG-потоки по камерам и JSON-статистику;
* позволяет ставить паузу/продолжить и менять conf на лету (кнопки в UI).

Запуск (в окружении с torch, для GPU — из ``.venv-rocm``):
    python apps/dashboard.py --model runs/.../weights/best.pt --device 0
    # затем открыть http://127.0.0.1:8000/

Зависимости — только стандартная библиотека + уже используемые cv2/numpy.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np

from multiview3d.anchor_azimuth import basis, load_coefficients, wrap_angle
from multiview3d.detect import YoloDetector
from multiview3d.fusion import FusionConfig
from multiview3d.pipeline import MultiViewPipeline, PipelineConfig
from multiview3d.truck_dataset import DEFAULT_REPAIRS, DEFAULT_ROOT, TruckDataset
from multiview3d.truck_pipeline import load_anchor_offset_views

HTML = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><title>Пеленга — трекинг</title>
<style>
 body{background:#111;color:#ddd;font-family:system-ui,sans-serif;margin:0}
 header{padding:10px 16px;background:#1b1b1b;display:flex;gap:16px;align-items:center}
 h1{font-size:16px;margin:0;font-weight:600}
 button{background:#2d6cdf;color:#fff;border:0;padding:6px 12px;border-radius:6px;cursor:pointer}
 button.alt{background:#444}
 .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:8px;padding:8px}
 .cell{background:#000;border:1px solid #333}
 .cell img{width:100%;display:block}
 .cap{padding:4px 8px;font-size:12px;color:#9ad}
 #stats{font-family:ui-monospace,monospace;font-size:12px;color:#9f9}
</style></head><body>
<header>
  <h1>Пеленга · трекинг в реальном времени</h1>
  <button onclick="ctl('toggle')">Пауза / продолжить</button>
  <button class="alt" onclick="ctl('conf_down')">conf −</button>
  <button class="alt" onclick="ctl('conf_up')">conf +</button>
  <span id="stats">…</span>
</header>
<div class="grid" id="grid"></div>
<script>
const cams = ["map","cam1","cam2","cam3","cam4","cam5"];
const titles = {map:"план X–Y (мир), зелёный — измерения, оранжевый — прогноз, белый — вперёд"};
const grid = document.getElementById('grid');
for (const c of cams) {
  const d = document.createElement('div'); d.className='cell';
  const cap = titles[c] || c;
  d.innerHTML = `<img src="/stream/${c}" alt="${c}"><div class="cap">${cap}</div>`;
  grid.appendChild(d);
}
async function ctl(action){ await fetch('/api/control?action='+action); }
async function tick(){
  try{
    const r = await fetch('/api/stats'); const s = await r.json();
    document.getElementById('stats').textContent =
      `кадр ${s.frame} · ${s.fps_now.toFixed(1)} fps · conf ${s.conf.toFixed(2)} · ` +
      `измерено ${(100*s.measured_share).toFixed(1)}% · ` +
      (s.position ? `X=${s.position[0].toFixed(2)} Y=${s.position[1].toFixed(2)} Z=${s.position[2].toFixed(2)} м` : 'нет 3D') +
      (s.paused ? ' · ПАУЗА' : '');
  }catch(e){}
}
setInterval(tick, 700); tick();
</script></body></html>"""


class Tracker:
    """Фоновый поток: чтение кадров, детекция, слияние, отрисовка."""

    def __init__(self, args):
        self.args = args
        self.ds = TruckDataset.discover(args.root, repairs_path=args.repairs or None)
        names = [n.strip() for n in args.class_names.split(",") if n.strip()]
        self.detector = YoloDetector(args.model, conf=args.conf, device=args.device,
                                     imgsz=args.imgsz, class_names=names or None)
        cfg = FusionConfig(association_mode="single_target",
                           max_reprojection_px=args.max_reprojection_px,
                           gate_mahalanobis=args.gate,
                           max_missing=args.max_missing)
        views = load_anchor_offset_views(args.anchor_views)
        if views:
            cfg.anchor_offset_views = {k: tuple(v) for k, v in views.items()}
        self.azimuth = None
        if Path(args.anchor_azimuth).exists():
            self.azimuth = load_coefficients(args.anchor_azimuth)

        frame0 = self.ds.common_frame_range()[0]
        cams = [self.ds.calibration(n, frame0) for n in self.ds.names]
        pipe_cfg = PipelineConfig(cameras=cams, fusion=cfg, fps=self.ds.fps,
                                  frame_start=frame0)
        self._pipe_cfg = pipe_cfg
        self._make_pipeline()

        self.caps = [cv2.VideoCapture(str(c.video)) for c in self.ds.cameras]
        self.frames = {}
        self.lock = threading.Lock()
        self.frame_idx = 0
        self.paused = False
        self.stop = False
        self.finished = False
        self.conf = float(args.conf)
        self.fps_now = 0.0
        self.conf_history = []

    def _make_pipeline(self):
        """Новый конвейер (нужен при перезапуске видео по кругу)."""
        self.pipe = MultiViewPipeline(self._pipe_cfg.cameras, config=self._pipe_cfg,
                                      camera_provider=self.ds.provider, fps=self.ds.fps)

    def _render_map(self, trail, forecast, position, status, t):
        """Отдельная панель: путь объекта в мировых координатах X–Y (план)."""
        W, H = 900, 640
        canvas = np.full((H, W, 3), (18, 20, 24), np.uint8)

        pts = []
        if trail:
            pts += [np.asarray(p[:2], dtype=float) for p, _ in trail]
        if forecast:
            pts += [np.asarray(q[:2], dtype=float) for q in forecast]
        if position is not None:
            pts.append(np.asarray(position[:2], dtype=float))

        if pts:
            P = np.vstack(pts)
            xmin, ymin = P.min(axis=0)
            xmax, ymax = P.max(axis=0)
        else:
            xmin = ymin = -100.0
            xmax = ymax = 100.0
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        span = max(xmax - xmin, ymax - ymin, 60.0) * 1.25

        # плавное расширение окна, чтобы карта не «дёргалась»
        if getattr(self, "_map_span", None) is None:
            self._map_span = span
            self._map_cx, self._map_cy = cx, cy
        else:
            self._map_span = max(self._map_span, span)
            self._map_cx = 0.7 * self._map_cx + 0.3 * cx
            self._map_cy = 0.7 * self._map_cy + 0.3 * cy

        scale = min((W - 80) / self._map_span, (H - 80) / self._map_span)

        def to_px(p):
            x = (float(p[0]) - self._map_cx) * scale + W / 2
            y = H / 2 - (float(p[1]) - self._map_cy) * scale
            return int(round(x)), int(round(y))

        # сетка с шагом, зависящим от масштаба
        step = 50.0 if self._map_span > 300 else (25.0 if self._map_span > 120 else 10.0)
        x0 = np.floor((self._map_cx - self._map_span / 2) / step) * step
        x1 = self._map_cx + self._map_span / 2
        while x0 <= x1:
            p0, p1 = to_px((x0, self._map_cy - self._map_span)), \
                to_px((x0, self._map_cy + self._map_span))
            cv2.line(canvas, p0, p1, (40, 44, 50), 1)
            cv2.putText(canvas, f"{x0:.0f}", (p0[0] + 2, H - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 130, 140), 1, cv2.LINE_AA)
            x0 += step
        y0 = np.floor((self._map_cy - self._map_span / 2) / step) * step
        y1 = self._map_cy + self._map_span / 2
        while y0 <= y1:
            p0, p1 = to_px((self._map_cx - self._map_span, y0)), \
                to_px((self._map_cx + self._map_span, y0))
            cv2.line(canvas, p0, p1, (40, 44, 50), 1)
            cv2.putText(canvas, f"{y0:.0f}", (6, p0[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 130, 140), 1, cv2.LINE_AA)
            y0 += step
        cv2.line(canvas, (W // 2, 0), (W // 2, H), (60, 66, 74), 1)
        cv2.line(canvas, (0, H // 2), (W, H // 2), (60, 66, 74), 1)
        cv2.putText(canvas, "X, м", (W - 60, H // 2 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 160, 170), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Y, м", (W // 2 + 8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 160, 170), 1, cv2.LINE_AA)

        # путь
        if trail:
            prev = None
            for p, st in trail:
                cur = to_px(p)
                if prev is not None:
                    if st == "MEASURED":
                        cv2.line(canvas, prev, cur, (20, 90, 20), 7, cv2.LINE_AA)
                        cv2.line(canvas, prev, cur, (80, 255, 80), 3, cv2.LINE_AA)
                    else:
                        cv2.line(canvas, prev, cur, (0, 80, 140), 5, cv2.LINE_AA)
                        cv2.line(canvas, prev, cur, (0, 175, 255), 2, cv2.LINE_AA)
                prev = cur
            cv2.circle(canvas, to_px(trail[0][0]), 5, (200, 200, 200), -1)
            cv2.putText(canvas, "старт", (to_px(trail[0][0])[0] + 8,
                                          to_px(trail[0][0])[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
        if forecast:
            prev = to_px(forecast[0])
            for i, q in enumerate(forecast):
                cur = to_px(q)
                if i % 2 == 0:
                    cv2.line(canvas, prev, cur, (245, 245, 245), 2, cv2.LINE_AA)
                prev = cur
        if position is not None:
            p = to_px(position)
            cv2.drawMarker(canvas, p, (80, 255, 80), cv2.MARKER_SQUARE, 16, 2)
            cv2.putText(canvas, f"({position[0]:.0f}, {position[1]:.0f})  {status}",
                        (p[0] + 12, p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (80, 255, 80), 1, cv2.LINE_AA)

        cv2.rectangle(canvas, (0, 0), (W, 30), (0, 0, 0), -1)
        cv2.putText(canvas, f"план X–Y (мир)   f={self.frame_idx}  t={t:6.2f} s",
                    (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 1, cv2.LINE_AA)
        return canvas

    def _render(self, name, img, dets, position, status, views, t,
                trail=None, forecast=None):
        for d in dets:
            if d.bbox is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in d.bbox]
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 190, 0), 2)
            cv2.putText(img, f"{d.score:.2f}", (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 190, 0), 1, cv2.LINE_AA)
        cam = self.ds.provider.calibrations_at(self.frame_idx, nearest=True).get(name)
        if cam is not None and trail:
            # Траектория: толстая зелёная — измерения, оранжевая — прогноз
            prev = None
            for p, st in trail:
                try:
                    uv = project(cam, p)
                except Exception:
                    prev = None
                    continue
                cur = (int(uv[0]), int(uv[1]))
                if prev is not None:
                    if st == "MEASURED":
                        cv2.line(img, prev, cur, (20, 90, 20), 8, cv2.LINE_AA)
                        cv2.line(img, prev, cur, (80, 255, 80), 4, cv2.LINE_AA)
                    else:
                        cv2.line(img, prev, cur, (0, 80, 140), 6, cv2.LINE_AA)
                        cv2.line(img, prev, cur, (0, 175, 255), 3, cv2.LINE_AA)
                prev = cur
            # Прогноз вперёд «призраком»: белая пунктирная линия
            if forecast:
                prev = None
                for i, q in enumerate(forecast):
                    try:
                        uv = project(cam, q)
                    except Exception:
                        prev = None
                        continue
                    cur = (int(uv[0]), int(uv[1]))
                    if prev is not None and i % 2 == 0:
                        cv2.line(img, prev, cur, (245, 245, 245), 2, cv2.LINE_AA)
                    prev = cur
        if position is not None and cam is not None:
            try:
                u, v = project(cam, position)
                cv2.drawMarker(img, (int(u), int(v)), (60, 200, 60),
                               cv2.MARKER_SQUARE, 18, 2)
                cv2.putText(img, f"3D {status}", (int(u) + 12, int(v) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 200, 60), 2, cv2.LINE_AA)
            except Exception:
                pass
        cv2.rectangle(img, (0, 0), (620, 52), (0, 0, 0), -1)
        cv2.putText(img, f"{name}  f={self.frame_idx}  t={t:6.2f} s  det={len(dets)}",
                    (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
        # легенда: что означают линии
        cv2.line(img, (10, 42), (38, 42), (60, 200, 60), 2)
        cv2.putText(img, "измерено", (44, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (200, 200, 200), 1, cv2.LINE_AA)
        cv2.line(img, (120, 42), (148, 42), (0, 160, 255), 2)
        cv2.putText(img, "прогноз", (154, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (200, 200, 200), 1, cv2.LINE_AA)
        cv2.line(img, (225, 42), (253, 42), (230, 230, 230), 1)
        cv2.putText(img, "вперёд", (259, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (200, 200, 200), 1, cv2.LINE_AA)
        return img

    def _apply_azimuth(self, last_points):
        """Поправка «от ракурса» к последней точке каждого трека (для показа)."""
        if not self.azimuth:
            return
        for track in self.pipe.all_tracks():
            pts = sorted(track.points, key=lambda p: p.timestamp)
            if len(pts) < 2:
                continue
            p = pts[-1]
            views = tuple(sorted(p.views or ()))
            key = "|".join(views)
            coef = self.azimuth.get(key)
            if coef is None or len(views) < 2 or p.frame is None:
                continue
            d = np.asarray(pts[-1].position) - np.asarray(pts[-2].position)
            if np.hypot(d[0], d[1]) < 1e-3:
                continue
            heading = float(np.arctan2(d[1], d[0]))
            cam = self.ds.provider.calibrations_at(p.frame, nearest=True).get(views[0])
            if cam is None:
                continue
            C = -np.asarray(cam.R, dtype=float).T @ np.asarray(cam.t, dtype=float).reshape(3)
            P = np.asarray(p.position, dtype=float)
            bearing = float(np.arctan2(P[1] - C[1], P[0] - C[0]))
            p.position = P - (basis(wrap_angle(bearing - heading)) @ coef)

    def _forecast(self, pts) -> list | None:
        """«Призрак» будущей траектории: экстраполяция по последней скорости."""
        horizon = float(self.args.forecast_s)
        if horizon <= 0 or len(pts) < 2:
            return None
        last = pts[-1]
        v = np.asarray(getattr(last, "velocity", np.zeros(3)), dtype=float)
        if not np.all(np.isfinite(v)) or float(np.hypot(v[0], v[1])) < 0.1:
            prev = np.asarray(pts[-2].position, dtype=float)
            cur = np.asarray(last.position, dtype=float)
            dt = max(float(last.timestamp) - float(pts[-2].timestamp), 1e-3)
            v = (cur - prev) / dt
        if float(np.hypot(v[0], v[1])) < 0.1:
            return None
        start = np.asarray(last.position, dtype=float)
        t0 = float(last.timestamp)
        steps = 12
        return [start + v * (horizon * i / steps) for i in range(1, steps + 1)]

    def loop(self):
        last_t = time.perf_counter()
        while not self.stop:
            if self.paused:
                time.sleep(0.05)
                continue
            t0 = time.perf_counter()
            imgs = []
            for cap in self.caps:
                ok, img = cap.read()
                imgs.append(img if ok else None)
            if any(im is None for im in imgs):
                if self.args.loop:
                    for cap in self.caps:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.frame_idx = 0
                    self._make_pipeline()
                    continue
                self.finished = True
                break
            t = self.frame_idx / self.ds.fps
            dets_by_cam = {}
            for cam, img in zip(self.ds.cameras, imgs):
                d = self.detector.detect(img)
                for det in d:
                    det.camera = cam.name
                    det.timestamp = t
                dets_by_cam[cam.name] = d
            result = self.pipe.step(t, dets_by_cam, frame=self.frame_idx)
            self._apply_azimuth(None)
            position, status, views = None, "-", []
            trail, forecast = None, None
            if result.tracks:
                pts = sorted(result.tracks[0].points, key=lambda p: p.timestamp)
                if pts:
                    last = pts[-1]
                    position, status, views = (np.asarray(last.position), last.status,
                                               list(last.views))
                    n = max(int(self.args.trail), 2)
                    trail = [(np.asarray(p.position, dtype=float), p.status)
                             for p in pts[-n:]]
                    forecast = self._forecast(pts)
            for cam, img in zip(self.ds.cameras, imgs):
                vis = self._render(cam.name, img, dets_by_cam.get(cam.name, []),
                                   position, status, views, t, trail, forecast)
                if self.args.preview_scale != 1.0:
                    h, w = vis.shape[:2]
                    vis = cv2.resize(vis, (int(w * self.args.preview_scale),
                                           int(h * self.args.preview_scale)),
                                     interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    with self.lock:
                        self.frames[cam.name] = buf.tobytes()
            try:
                mp = self._render_map(trail, forecast, position, status, t)
                ok, buf = cv2.imencode(".jpg", mp, [cv2.IMWRITE_JPEG_QUALITY, 82])
                if ok:
                    with self.lock:
                        self.frames["map"] = buf.tobytes()
            except Exception as exc:
                print(f"карта: {exc}", file=sys.stderr)
            dt = time.perf_counter() - t0
            self.fps_now = 1.0 / max(dt, 1e-6)
            self.frame_idx += 1
            target = 1.0 / self.args.target_fps if self.args.target_fps else 0.0
            if target and time.perf_counter() - t0 < target:
                time.sleep(target - (time.perf_counter() - t0))
            last_t = time.perf_counter()

    def stats(self):
        cov = self.pipe.coverage or {}
        frames = max(int(cov.get("frames", 0)), 1)
        pos = None
        if self.pipe.last_result and self.pipe.last_result.tracks:
            pts = self.pipe.last_result.tracks[0].points
            if pts:
                pos = [round(float(v), 3) for v in pts[-1].position]
        return {"frame": self.frame_idx, "fps_now": round(self.fps_now, 2),
                "conf": self.conf, "paused": self.paused,
                "finished": self.finished,
                "measured_share": round(float(cov.get("measured", 0)) / frames, 3),
                "measured_frames": int(cov.get("measured", 0)),
                "position": pos}


def project(cam, P):
    from multiview3d.camera import project_point
    return project_point(cam, np.asarray(P, dtype=float))


def make_handler(tracker: Tracker):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/":
                body = HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif u.path.startswith("/stream/"):
                cam = u.path.split("/")[-1]
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        with tracker.lock:
                            buf = tracker.frames.get(cam)
                        if buf is None:
                            time.sleep(0.05)
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(buf)}\r\n\r\n".encode())
                        self.wfile.write(buf + b"\r\n")
                        time.sleep(1.0 / max(tracker.fps_now, 1.0))
                except (BrokenPipeError, ConnectionResetError,
                        ConnectionAbortedError, OSError):
                    return
            elif u.path == "/api/stats":
                body = json.dumps(tracker.stats()).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif u.path == "/api/control":
                action = parse_qs(u.query).get("action", [""])[0]
                if action == "toggle":
                    tracker.paused = not tracker.paused
                elif action == "conf_up":
                    tracker.conf = min(tracker.conf + 0.05, 0.95)
                    tracker.detector.conf = tracker.conf
                elif action == "conf_down":
                    tracker.conf = max(tracker.conf - 0.05, 0.05)
                    tracker.detector.conf = tracker.conf
                body = json.dumps({"ok": True, "paused": tracker.paused,
                                   "conf": tracker.conf}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_error(404)

    return Handler


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Веб-пульт трекинга «Пеленга»")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--repairs", default=str(DEFAULT_REPAIRS))
    ap.add_argument("--model", default="runs/detect/out/train/scene_v1/weights/best.pt")
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--class-names", default="truck",
                    help="для дообученных моделей — 'truck'; для COCO — 'truck,car,bus,train'")
    ap.add_argument("--anchor-views", default="datasets/truck/anchor_offset_views.json")
    ap.add_argument("--anchor-azimuth", default="datasets/truck/anchor_azimuth.json")
    ap.add_argument("--max-reprojection-px", type=float, default=50.0)
    ap.add_argument("--gate", type=float, default=10.0)
    ap.add_argument("--target-fps", type=float, default=15.0,
                    help="ограничение темпа (0 — без ограничения)")
    ap.add_argument("--preview-scale", type=float, default=1.0,
                    help="масштаб превью (0.5 — вдвое меньше и быстрее кодирование)")
    ap.add_argument("--loop", action="store_true",
                    help="проигрывать видео по кругу (демо)")
    ap.add_argument("--trail", type=int, default=80,
                    help="сколько последних 3D-точек рисовать как траекторию")
    ap.add_argument("--forecast-s", type=float, default=2.0,
                    help="горизонт «призрачного» прогноза вперёд, с (0 — выключить)")
    ap.add_argument("--max-missing", type=int, default=30,
                    help="сколько кадров без измерения продолжать прогноз (Калман)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args(argv)

    tracker = Tracker(args)
    th = threading.Thread(target=tracker.loop, daemon=True)
    th.start()
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(tracker))
    print(f"Пульт: http://{args.host}:{args.port}/  (Ctrl+C — остановить)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        tracker.stop = True
        th.join(timeout=2.0)
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
