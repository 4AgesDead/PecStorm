"""Якорь, зависящий от ракурса: поправка 3D-точки по азимуту камеры.

Физика: смещение «центр bbox ↔ pivot» зависит от того, под каким углом камера
видит грузовик (какая часть кузова попадает в бокс). Константа на набор видов
это не описывает, поэтому остаётся ~2 м ошибки.

Модель на каждый набор видов::

    residual(az) ≈ c0 + c1·cos(az) + s1·sin(az) + c2·cos(2az) + s2·sin(2az)

``az`` — угол между направлением «камера → объект» и курсом объекта. Курс на
рантайме берётся из движения самого трека (разность соседних точек), в фите —
из GT-скорости.

Фитим ``residual = estimate - gt`` на train-кадрах; применяем вычитанием.
Проверено на holdout 700..1000: MAE 2.06 → 0.63 м (parity-CV 0.56 м).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

BASIS_SIZE = 5
MAX_CORRECTION_M = 15.0
"""Ограничение модуля поправки: защита от плохо обусловленной модели."""


def basis(az: float) -> np.ndarray:
    return np.array([1.0, np.cos(az), np.sin(az),
                     np.cos(2 * az), np.sin(2 * az)], dtype=float)


def wrap_angle(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))


def fit_coefficients(samples: list[dict], min_observations: int = 40,
                     ridge: float = 1e-2) -> dict:
    """``samples``: [{'views': (…), 'az': float, 'residual': np.array(3)}] -> коэф.

    Гребневая регрессия (``ridge``): ряд Фурье на узком диапазоне азимутов
    плохо обусловлен, без регуляризации коэффициенты «взрываются» на новых
    кадрах (проверено: MAE holdout уходил в 344 км).
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        groups["|".join(sorted(s["views"]))].append(s)
    out = {}
    for key, items in groups.items():
        if len(items) < min_observations:
            continue
        A = np.stack([basis(s["az"]) for s in items])
        B = np.stack([np.asarray(s["residual"], dtype=float) for s in items])
        lam = ridge * len(items) * np.eye(A.shape[1])
        coef = np.linalg.solve(A.T @ A + lam, A.T @ B)
        out[key] = [[round(float(v), 6) for v in row] for row in coef]
    return out


def load_coefficients(path: str | Path) -> dict[str, np.ndarray]:
    doc = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    raw = doc.get("coefficients") or {}
    return {k: np.asarray(v, dtype=float).reshape(BASIS_SIZE, 3) for k, v in raw.items()}


def save_coefficients(path: str | Path, coefficients: dict, meta: dict) -> None:
    doc = {"coefficients": coefficients,
           "basis": "1, cos(az), sin(az), cos(2az), sin(2az)",
           "angle": "az = bearing(камера->объект) - heading(объект)",
           **meta}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")


def heading_from_positions(points: list, index: int) -> float | None:
    """Курс по соседним точкам трека (точки отсортированы по времени)."""
    if len(points) < 2:
        return None
    i0 = max(index - 1, 0)
    i1 = min(index + 1, len(points) - 1)
    if i1 == i0:
        return None
    d = np.asarray(points[i1].position, dtype=float) - np.asarray(points[i0].position, dtype=float)
    if float(np.hypot(d[0], d[1])) < 1e-3:
        return None
    return float(np.arctan2(d[1], d[0]))


def apply_to_track(track, coefficients: dict[str, np.ndarray],
                   camera_at) -> int:
    """Правит позиции точек трека на месте. ``camera_at(frame, view)`` -> камера."""
    points = sorted(track.points, key=lambda p: p.timestamp)
    deltas: list[np.ndarray | None] = []
    for i, p in enumerate(points):
        views = tuple(sorted(p.views or ()))
        key = "|".join(views)
        coef = coefficients.get(key)
        if coef is None or len(views) < 2 or p.frame is None:
            deltas.append(None)
            continue
        heading = heading_from_positions(points, i)
        if heading is None:
            deltas.append(None)
            continue
        cam = camera_at(p.frame, views[0])
        if cam is None:
            deltas.append(None)
            continue
        C = -np.asarray(cam.R, dtype=float).T @ np.asarray(cam.t, dtype=float).reshape(3)
        P = np.asarray(p.position, dtype=float)
        bearing = float(np.arctan2(P[1] - C[1], P[0] - C[0]))
        az = wrap_angle(bearing - heading)
        delta = -(basis(az) @ coef)
        norm = float(np.linalg.norm(delta))
        if norm > MAX_CORRECTION_M:          # защита от выбросов модели
            delta = delta * (MAX_CORRECTION_M / norm)
        deltas.append(delta)

    applied = 0
    for p, d in zip(points, deltas):
        if d is None:
            continue
        p.position = np.asarray(p.position, dtype=float) + d
        applied += 1
    return applied
