"""Отрисовка пути и панели плана X–Y — общая для пульта и офлайн-видео.

История — список ``(t, position, status)``. Траектория строится одинаково:

* сглаживание Савицкого–Голея (локальный полином 2-й степени);
* разрыв при пропадании > ``max_gap_s`` (серый пунктир-соединитель);
* разрыв при скачке трека > ``jump_m`` (без соединителя);
* достройка после последнего измерения не дальше ``bridge_s``;
* прогноз вперёд по средней скорости последнего участка измерений.

Функции не знают про HTTP/камеры, поэтому их использует и веб-пульт
(``apps/dashboard.py``), и рендер MP4 (``tools/render_tracking_video.py``).
"""

from __future__ import annotations

import cv2
import numpy as np

JUMP = object()                     # разрыв из-за скачка трека
_SG_CACHE: dict[int, np.ndarray] = {}

STYLES_VIDEO = {"measured": ((20, 90, 20), 8, (80, 255, 80), 4),
                "predicted": ((0, 80, 140), 6, (0, 175, 255), 3)}
STYLES_MAP = {"measured": ((20, 90, 20), 8, (80, 255, 80), 4),
              "predicted": ((0, 80, 140), 6, (0, 175, 255), 3)}


def sg_kernel(k: int, order: int = 2) -> np.ndarray:
    """Коэффициенты Савицкого–Голея: сглаживание без сдвига фазы."""
    k = max(int(k), 1)
    if k not in _SG_CACHE:
        x = np.arange(-k, k + 1, dtype=float)
        a = np.vander(x, min(order, 2 * k) + 1, increasing=True)
        _SG_CACHE[k] = np.linalg.pinv(a)[0]
    return _SG_CACHE[k]


def smooth_run(arr: np.ndarray, k: int) -> np.ndarray:
    """Сгладить участок пути локальным полиномом 2-й степени по окну ±k.

    Края продолжаются линейно, поэтому равномерное движение не «отстаёт».
    """
    k = int(k)
    if k <= 0 or len(arr) < 5:
        return arr
    k = min(k, (len(arr) - 1) // 2)
    if k <= 0:
        return arr
    n = len(arr)
    pad = np.empty((n + 2 * k, arr.shape[1]), dtype=float)
    pad[k:k + n] = arr
    for i in range(k):
        pad[k - 1 - i] = 2.0 * arr[0] - arr[i + 1]
        pad[k + n + i] = 2.0 * arr[-1] - arr[-2 - i]
    kern = sg_kernel(k)
    out = np.empty_like(arr)
    for d in range(arr.shape[1]):
        out[:, d] = np.convolve(pad[:, d], kern, mode="valid")
    return out


def dashed_line(canvas, a, b, color, thick=2, dash=9, step=15) -> None:
    """Пунктир между двумя пиксельными точками."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    dist = float(np.linalg.norm(b - a))
    if dist < 1.0:
        return
    u = (b - a) / dist
    s = 0.0
    while s < dist:
        e = min(s + dash, dist)
        p0 = tuple(np.round(a + u * s).astype(int))
        p1 = tuple(np.round(a + u * e).astype(int))
        cv2.line(canvas, p0, p1, color, thick, cv2.LINE_AA)
        s += step


def path_runs(trail):
    """Непрерывные участки пути: список ``(статус, [точки])``."""
    runs: list = []
    cur: list = []
    st_cur = None
    for item in trail:
        if item is None or item is JUMP:
            if cur:
                runs.append((st_cur, cur))
            cur, st_cur = [], None
            continue
        p, st = item
        if cur and st != st_cur:            # смена статуса — новый отрезок
            runs.append((st_cur, cur))
            cur = [cur[-1]]                 # сшить без визуального разрыва
        cur.append(np.asarray(p, dtype=float))
        st_cur = st
    if cur:
        runs.append((st_cur, cur))
    return runs


def gap_pairs(trail):
    """Пары точек до/после пропадания (None), но не после скачка трека."""
    prev, pending = None, False
    for item in trail:
        if item is None:
            pending = True
            continue
        if item is JUMP:
            pending, prev = False, None
            continue
        p = np.asarray(item[0], dtype=float)
        if pending and prev is not None:
            yield prev, p
        pending, prev = False, p


def draw_path(canvas, trail, to_pixels, styles, max_gap_m=120.0,
              gap_thick=2) -> None:
    """Нарисовать путь полилиниями (ореол + ядро) вместо тысяч cv2.line."""
    for st, pts in path_runs(trail):
        if len(pts) < 2:
            continue
        uv = np.asarray(to_pixels(np.stack(pts)), dtype=float)
        halo, hth, core, cth = styles["measured" if st == "MEASURED"
                                       else "predicted"]
        ok = np.isfinite(uv).all(axis=1)
        i = 0
        while i < len(uv):
            if not ok[i]:
                i += 1
                continue
            j = i
            while j < len(uv) and ok[j]:
                j += 1
            if j - i >= 2:
                poly = np.round(uv[i:j]).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(canvas, [poly], False, halo, hth, cv2.LINE_AA)
                cv2.polylines(canvas, [poly], False, core, cth, cv2.LINE_AA)
            i = j
    # пропадание измерения: серый пунктир «здесь данных нет», путь не рвётся
    for a_world, b_world in gap_pairs(trail):
        if float(np.linalg.norm(b_world - a_world)) > max_gap_m:
            continue
        uv = np.asarray(to_pixels(np.stack([a_world, b_world])), dtype=float)
        if not np.isfinite(uv).all():
            continue
        dashed_line(canvas, uv[0], uv[1], (145, 145, 145), gap_thick)


def build_trail(history, n: int, smooth: int = 5,
                max_gap_s: float = 0.2, jump_m: float = 10.0):
    """История → линия: разрывы при пропаданиях и сглаживание SG."""
    n = max(int(n), 2)
    h = history[-n:]
    if not h:
        return None
    seq: list = []
    for i, (t, p, st) in enumerate(h):
        gap = i and (t - h[i - 1][0]) > max_gap_s
        jump = i and float(np.linalg.norm(np.asarray(p) - np.asarray(h[i - 1][1]))) > jump_m
        if gap or jump:
            # пропадание/телепорт: серый пунктир-соединитель, путь не рвётся
            seq.append(None)
        seq.append((np.asarray(p, dtype=float), st))
    out: list = []
    i = 0
    k = int(smooth)
    while i < len(seq):
        if not isinstance(seq[i], tuple):
            out.append(seq[i])
            i += 1
            continue
        j = i
        while j < len(seq) and isinstance(seq[j], tuple):
            j += 1
        run = seq[i:j]
        arr = np.stack([p for p, _ in run])
        sm = smooth_run(arr, k)
        out.extend([(sm[m], run[m][1]) for m in range(len(run))])
        i = j
    return out


def measured_run(history, max_gap_s: float = 0.2):
    """Последний непрерывный отрезок ИЗМЕРЕНИЙ: (t0, p0, t1, p1)."""
    measured = [h for h in history if h[2] == "MEASURED"]
    if len(measured) < 5:
        return None
    run = [measured[-1]]
    for prev in reversed(measured[:-1]):
        if run[-1][0] - prev[0] > max_gap_s:
            break
        run.append(prev)
    run.reverse()
    if len(run) < 5:
        return None
    t0, p0, _ = run[0]
    t1, p1, _ = run[-1]
    if t1 - t0 <= 0.05:
        return None
    return t0, np.asarray(p0, dtype=float), t1, np.asarray(p1, dtype=float)


def mean_velocity(history):
    """Средняя скорость по последнему участку измерений: (v, t, p)."""
    run = measured_run(history)
    if run is None:
        return None
    t0, p0, t1, p1 = run
    v = (p1 - p0) / (t1 - t0)
    if float(np.hypot(v[0], v[1])) < 0.3:
        return None
    return v, t1, p1


def extrapolate(history, t: float, bridge_s: float):
    """Достроить путь после пропадания: строго от последнего ИЗМЕРЕНИЯ."""
    mv = mean_velocity(history)
    if mv is None:
        return None
    v, t_last, p_last = mv
    dt = float(t) - t_last
    if dt <= 0 or dt > float(bridge_s):
        return None
    return np.asarray(p_last, dtype=float) + v * dt


def forecast(history, horizon_s: float, bridge_s: float, steps: int = 16):
    """«Призрак» будущего: продолжение по средней скорости измерений."""
    horizon = float(horizon_s)
    mv = mean_velocity(history)
    if horizon <= 0 or mv is None or not history:
        return None
    v, t_last, _ = mv
    if history[-1][0] - t_last > float(bridge_s):
        return None
    start = np.asarray(history[-1][1], dtype=float)
    return [start + v * (horizon * i / steps) for i in range(1, steps + 1)]


def map_window(points, view=None):
    """Автоподбор окна карты: (span, cx, cy). ``view`` — предыдущее окно."""
    pts = [np.asarray(p[:2], dtype=float) for p in points]
    if len(pts) >= 5:
        p = np.vstack(pts)
        # 2–98 %: случайный выброс триангуляции не должен «раздувать» карту
        lo = np.percentile(p, 2, axis=0)
        hi = np.percentile(p, 98, axis=0)
        cx, cy = (lo + hi) / 2
        span = max(float(hi[0] - lo[0]), float(hi[1] - lo[1]), 60.0) * 1.3
    elif pts:
        p = np.vstack(pts)
        cx, cy = p.mean(axis=0)
        span = max(float(np.ptp(p[:, 0])), float(np.ptp(p[:, 1])), 60.0) * 1.3
    else:
        cx, cy, span = 0.0, 0.0, 120.0
    if view is None:
        return float(span), float(cx), float(cy)
    pspan, pcx, pcy = view
    # окно расширяется сразу, а сжимается плавно — карта не «дёргается»
    span = max(0.85 * pspan + 0.15 * span, span)
    return float(span), 0.8 * pcx + 0.2 * float(cx), 0.8 * pcy + 0.2 * float(cy)


def render_map_panel(trail, forecast_points, position, status, t, frame_idx,
                     view=None, size=(900, 640), gt_points=None,
                     gt_position=None):
    """Панель «план X–Y (мир)»: сетка, путь, прогноз, текущая точка.

    ``gt_points``/``gt_position`` — эталон (GT): жёлтая линия и крест.
    Возвращает ``(canvas, view)``; ``view`` нужно передать при следующем
    вызове, чтобы окно не дёргалось.
    """
    w, h = size
    points = [item[0] for item in (trail or []) if isinstance(item, tuple)]
    if position is not None:
        points.append(position)
    if gt_points:
        points.extend(gt_points)
    if gt_position is not None:
        points.append(gt_position)
    span, cx, cy = map_window(points, view)
    scale = min((w - 80) / span, (h - 80) / span)
    canvas = np.full((h, w, 3), (18, 20, 24), np.uint8)

    def to_px(p):
        x = (float(p[0]) - cx) * scale + w / 2
        y = h / 2 - (float(p[1]) - cy) * scale
        return int(round(x)), int(round(y))

    def to_px_batch(p):
        p = np.asarray(p, dtype=float)
        x = (p[:, 0] - cx) * scale + w / 2
        y = h / 2 - (p[:, 1] - cy) * scale
        return np.stack((x, y), axis=1)

    step = 50.0 if span > 300 else (25.0 if span > 120 else 10.0)
    x0 = np.floor((cx - span / 2) / step) * step
    x1 = cx + span / 2
    while x0 <= x1:
        p0, p1 = to_px((x0, cy - span)), to_px((x0, cy + span))
        cv2.line(canvas, p0, p1, (40, 44, 50), 1)
        cv2.putText(canvas, f"{x0:.0f}", (p0[0] + 2, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 130, 140), 1, cv2.LINE_AA)
        x0 += step
    y0 = np.floor((cy - span / 2) / step) * step
    y1 = cy + span / 2
    while y0 <= y1:
        p0, p1 = to_px((cx - span, y0)), to_px((cx + span, y0))
        cv2.line(canvas, p0, p1, (40, 44, 50), 1)
        cv2.putText(canvas, f"{y0:.0f}", (6, p0[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 130, 140), 1, cv2.LINE_AA)
        y0 += step
    cv2.line(canvas, (w // 2, 0), (w // 2, h), (60, 66, 74), 1)
    cv2.line(canvas, (0, h // 2), (w, h // 2), (60, 66, 74), 1)
    cv2.putText(canvas, "X, м", (w - 60, h // 2 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 160, 170), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Y, м", (w // 2 + 8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 160, 170), 1, cv2.LINE_AA)

    # эталон: жёлтая линия под оценкой, чтобы не перекрывать её
    if gt_points:
        gp = np.asarray(gt_points, dtype=float)
        if gp.size:
            uv = to_px_batch(gp[:, :2])
            ok = np.isfinite(uv).all(axis=1)
            i = 0
            while i < len(uv):
                if not ok[i]:
                    i += 1
                    continue
                j = i
                while j < len(uv) and ok[j]:
                    j += 1
                if j - i >= 2:
                    poly = np.round(uv[i:j]).astype(np.int32).reshape(-1, 1, 2)
                    cv2.polylines(canvas, [poly], False, (0, 220, 255), 2,
                                  cv2.LINE_AA)
                i = j

    if trail:
        draw_path(canvas, trail, to_px_batch, STYLES_MAP)
        first = next((np.asarray(it[0]) for it in trail if isinstance(it, tuple)),
                     None)
        if first is not None:
            fp = to_px(first)
            cv2.circle(canvas, fp, 5, (200, 200, 200), -1)
            cv2.putText(canvas, "старт", (fp[0] + 8, fp[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1,
                        cv2.LINE_AA)
    if forecast_points:
        dashed_line(canvas, to_px(forecast_points[0]), to_px(forecast_points[-1]),
                    (245, 245, 245), 2, dash=7, step=14)
    if position is not None:
        p = to_px(position)
        col = (80, 255, 80) if status == "MEASURED" else (0, 175, 255)
        cv2.drawMarker(canvas, p, col, cv2.MARKER_SQUARE, 16, 2)
        cv2.putText(canvas, f"({position[0]:.0f}, {position[1]:.0f})  {status}",
                    (p[0] + 12, p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    col, 1, cv2.LINE_AA)
    if gt_position is not None:
        g = to_px(gt_position)
        cv2.drawMarker(canvas, g, (0, 220, 255), cv2.MARKER_CROSS, 18, 2)
        cv2.putText(canvas, "эталон", (g[0] + 10, g[1] + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1,
                    cv2.LINE_AA)

    cv2.rectangle(canvas, (0, 0), (w, 30), (0, 0, 0), -1)
    cv2.putText(canvas, f"план X–Y (мир)   f={frame_idx}  t={t:6.2f} s",
                (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 1,
                cv2.LINE_AA)
    cv2.putText(canvas, "серый пунктир — разрыв данных", (560, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1, cv2.LINE_AA)
    if gt_points:
        cv2.putText(canvas, "жёлтый — эталон", (390, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
    return canvas, (span, cx, cy)


def draw_legend(img, scale: float = 1.0) -> None:
    """Легенда линий, как на пульте (``scale`` — под размер кадра)."""
    s = max(float(scale), 0.5)

    def pt(x, y):
        return int(round(x * s)), int(round(y * s))

    thick = max(1, int(round(2 * s)))
    cv2.line(img, pt(10, 42), pt(38, 42), (60, 200, 60), thick)
    cv2.putText(img, "измерено", pt(44, 46), cv2.FONT_HERSHEY_SIMPLEX,
                0.42 * s, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.line(img, pt(120, 42), pt(148, 42), (0, 160, 255), thick)
    cv2.putText(img, "прогноз", pt(154, 46), cv2.FONT_HERSHEY_SIMPLEX,
                0.42 * s, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.line(img, pt(225, 42), pt(253, 42), (230, 230, 230), max(1, int(round(s))))
    cv2.putText(img, "вперёд", pt(259, 46), cv2.FONT_HERSHEY_SIMPLEX,
                0.42 * s, (200, 200, 200), 1, cv2.LINE_AA)
    dash_end = int(round(360 * s))
    for x0 in range(int(round(330 * s)), dash_end, max(1, int(round(8 * s)))):
        cv2.line(img, (x0, int(round(42 * s))),
                 (min(x0 + max(1, int(round(4 * s))), dash_end), int(round(42 * s))),
                 (145, 145, 145), max(1, int(round(s))), cv2.LINE_AA)
    cv2.putText(img, "разрыв", pt(366, 46), cv2.FONT_HERSHEY_SIMPLEX,
                0.42 * s, (200, 200, 200), 1, cv2.LINE_AA)
