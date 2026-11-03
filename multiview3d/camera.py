"""Геометрия проекции/триангуляции: лучи, искажения, эпиполярные метрики."""

from __future__ import annotations

import numpy as np
import cv2

from .types import CameraCalibration


def _distort(K: np.ndarray, dist: np.ndarray, x: float, y: float) -> tuple[float, float]:
    """Нормализованные координаты камеры -> искажённые координаты изображения."""
    if dist is None or np.allclose(dist, 0.0):
        return x, y
    d = np.asarray(dist, dtype=float).ravel()
    k1 = d[0] if d.size > 0 else 0.0
    k2 = d[1] if d.size > 1 else 0.0
    p1 = d[2] if d.size > 2 else 0.0
    p2 = d[3] if d.size > 3 else 0.0
    k3 = d[4] if d.size > 4 else 0.0
    k4 = d[5] if d.size > 5 else 0.0
    k5 = d[6] if d.size > 6 else 0.0
    k6 = d[7] if d.size > 7 else 0.0

    r2 = x * x + y * y
    radial = (1.0 + k1 * r2 + k2 * r2**2 + k3 * r2**3) / (
        1.0 + k4 * r2 + k5 * r2**2 + k6 * r2**3
    )
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return fx * xd + cx, fy * yd + cy


def _project_with_pose(K: np.ndarray, dist: np.ndarray, x_cam: np.ndarray) -> np.ndarray:
    """Пиксели для точек в системе камеры (OpenCV: x вправо, y вниз, z вперёд)."""
    x = x_cam[:, 0] / x_cam[:, 2]
    y = x_cam[:, 1] / x_cam[:, 2]
    if dist is None or np.allclose(dist, 0.0):
        return K @ np.vstack((x, y, np.ones_like(x)))
    u = np.empty(x.size)
    v = np.empty(x.size)
    for i in range(x.size):
        u[i], v[i] = _distort(K, dist, float(x[i]), float(y[i]))
    return np.vstack((u, v))


def project_point(cam: CameraCalibration, X: np.ndarray) -> np.ndarray:
    """Мировая точка X (3,) -> пиксель (u, v) с учётом искажений."""
    X = np.asarray(X, dtype=float).reshape(3)
    R = np.asarray(cam.R, dtype=float)
    t = np.asarray(cam.t, dtype=float).reshape(3)
    x_cam = R @ X + t
    if x_cam[2] <= 1e-9:
        raise ValueError("точка лежит не перед камерой (z <= 0)")
    uv = _project_with_pose(cam.K, cam.dist, x_cam.reshape(1, 3))
    return uv[:, 0]


def project_points(cam: CameraCalibration, points: np.ndarray) -> np.ndarray:
    """Мировые точки (N,3) -> пиксели (N,2). Не перед камерой -> NaN."""
    P = np.asarray(points, dtype=float).reshape(-1, 3)
    R = np.asarray(cam.R, dtype=float)
    t = np.asarray(cam.t, dtype=float).reshape(3)
    x_cam = P @ R.T + t
    out = np.full((P.shape[0], 2), np.nan)
    good = x_cam[:, 2] > 1e-9
    if good.any():
        out[good] = _project_with_pose(cam.K, cam.dist, x_cam[good]).T
    return out


def undistort_pixels(cam: CameraCalibration, pixels: np.ndarray) -> np.ndarray:
    """Искажённые пиксели (N,2) -> недисторсированные пиксели (N,2)."""
    px = np.asarray(pixels, dtype=float).reshape(-1, 1, 2)
    if cam.dist is None or np.allclose(cam.dist, 0.0):
        return px.reshape(-1, 2).copy()
    ideal = cv2.undistortPoints(px, cam.K, cam.dist, P=cam.K)
    return ideal.reshape(-1, 2)


def ideal_camera(cam: CameraCalibration) -> CameraCalibration:
    """Камера без искажений с той же матрицей K (для эпиполярной геометрии)."""
    return CameraCalibration(
        name=cam.name,
        K=cam.K.copy(),
        dist=np.zeros(5, dtype=float),
        R=cam.R.copy(),
        t=cam.t.copy(),
        width=cam.width,
        height=cam.height,
    )


def back_project_ray(cam: CameraCalibration, pixel: np.ndarray) -> np.ndarray:
    """Единичный луч камеры через пиксель в мировых координатах."""
    px = np.asarray(pixel, dtype=float).reshape(1, 1, 2)
    norm = cv2.undistortPoints(px, cam.K, cam.dist)  # нормализованные координаты (z=1)
    d = norm[0, 0]
    dir_cam = np.array([d[0], d[1], 1.0], dtype=float)
    R = np.asarray(cam.R, dtype=float)
    dir_world = R.T @ dir_cam
    return dir_world / np.linalg.norm(dir_world)


def triangulate_rays(rays: list[np.ndarray], origins: list[np.ndarray]) -> np.ndarray:
    """DLT-триангуляция по лучам: минимум суммы квадратов расстояний до лучей.

    Каждая лучевая точка удовлетворяет ``(I - d d^T)(X - o) = 0``; решается
    линейная система ``A X = sum((I - d_i d_i^T) o_i)``, ``A = sum(I - d_i d_i^T)``.
    """
    A = np.zeros((3, 3), dtype=float)
    b = np.zeros(3, dtype=float)
    for o, d in zip(origins, rays):
        o = np.asarray(o, dtype=float).reshape(3)
        d = np.asarray(d, dtype=float).reshape(3)
        d = d / max(np.linalg.norm(d), 1e-12)
        M = np.eye(3) - np.outer(d, d)
        A += M
        b += M @ o
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]


def refine_point(
    cams: list[CameraCalibration],
    pixels: list[np.ndarray],
    X0: np.ndarray,
    iterations: int = 12,
    lambda0: float = 1e-3,
) -> np.ndarray:
    """Уточнение 3D-точки Левенбергом–Марквардом в метрике репроекции (пиксели, с искажениями)."""
    X = np.asarray(X0, dtype=float).reshape(3).copy()
    lam = lambda0
    prev_cost = None

    def uv_of(cam: CameraCalibration, Xc: np.ndarray) -> np.ndarray:
        try:
            return project_point(cam, Xc)
        except ValueError:
            return np.array([np.nan, np.nan])

    def residual(Xc: np.ndarray) -> np.ndarray:
        res = []
        for cam, px in zip(cams, pixels):
            uv = uv_of(cam, Xc)
            res.extend([uv[0] - px[0], uv[1] - px[1]])
        return np.array(res, dtype=float)

    cost = float(np.sum(residual(X) ** 2))
    for _ in range(iterations):
        step = 1e-4
        J = np.zeros((2 * len(cams), 3), dtype=float)
        r = residual(X)
        for i, cam in enumerate(cams):
            uv0 = uv_of(cam, X)
            for k in range(3):
                Xd = X.copy()
                Xd[k] += step
                ud = uv_of(cam, Xd)
                # Якобиан невязок: d(uv - px)/dX = d uv / dX.
                J[2 * i, k] = (ud[0] - uv0[0]) / step
                J[2 * i + 1, k] = (ud[1] - uv0[1]) / step
        JTJ = J.T @ J
        JTr = J.T @ r
        try:
            dx = np.linalg.solve(JTJ + lam * np.eye(3), -JTr)
        except np.linalg.LinAlgError:
            break
        Xn = X + dx
        try:
            cn = float(np.sum(residual(Xn) ** 2))
        except Exception:
            break
        if np.isfinite(cn) and cn < cost:
            X, cost = Xn, cn
            lam = max(lam * 0.3, 1e-8)
        else:
            lam = min(lam * 10.0, 1e6)
        if prev_cost is not None and abs(prev_cost - cost) < 1e-12:
            break
        prev_cost = cost
    return X


def reprojection_error(cam: CameraCalibration, X: np.ndarray, pixel: np.ndarray) -> float:
    """Евклидова ошибка репроекции в пикселях."""
    try:
        uv = project_point(cam, X)
    except ValueError:
        return float("inf")
    px = np.asarray(pixel, dtype=float).reshape(2)
    return float(np.linalg.norm(uv - px))


def fundamental_matrix(c1: CameraCalibration, c2: CameraCalibration) -> np.ndarray:
    """Фундаментальная матрица для пар недисторсированных камер."""
    i1, i2 = ideal_camera(c1), ideal_camera(c2)
    return _fundamental(i1, i2)


def _fundamental(c1: CameraCalibration, c2: CameraCalibration) -> np.ndarray:
    R = np.asarray(c2.R, dtype=float) @ np.asarray(c1.R, dtype=float).T
    t = np.asarray(c2.t, dtype=float) - R @ np.asarray(c1.t, dtype=float)
    E = _skew(t) @ R
    K1 = np.asarray(c1.K, dtype=float)
    K2 = np.asarray(c2.K, dtype=float)
    # Строчная конвенция: u2^T F u1 = 0 => F = K2^-T @ E @ K1^-1.
    F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
    return F / np.linalg.norm(F)


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=float
    )


def epipolar_distance_batch(
    c1: CameraCalibration, c2: CameraCalibration, pixels1: np.ndarray, pixels2: np.ndarray
) -> np.ndarray:
    """Расстояния точки пикселя2 от эпиполярной линии пикселя1 в пикселях камеры 2.

    Искажения снимаются (недисторсированные пиксели), расстояние нормализуется на
    масштаб эпиполярной линии, что даёт пиксельную метрику.
    """
    p1 = undistort_pixels(c1, np.asarray(pixels1, dtype=float).reshape(-1, 2))
    p2 = undistort_pixels(c2, np.asarray(pixels2, dtype=float).reshape(-1, 2))
    F = fundamental_matrix(c1, c2)
    n = min(p1.shape[0], p2.shape[0])
    out = np.full(n, np.inf)
    for i in range(n):
        line = F @ np.array([p1[i, 0], p1[i, 1], 1.0])
        norm = np.hypot(line[0], line[1])
        if norm < 1e-12:
            continue
        d = abs(line @ np.array([p2[i, 0], p2[i, 1], 1.0])) / norm
        out[i] = d
    return out


def epipolar_distance_matrix(
    c1: CameraCalibration, c2: CameraCalibration, pixels1: np.ndarray, pixels2: np.ndarray
) -> np.ndarray:
    """Полная матрица ``(N1, N2)``: расстояние каждого пикселя ``pixels2`` до
    эпиполярной линии, построенной по каждому пикселю ``pixels1`` (в пикселях c2).

    Пиксели недисторсируются; расстояние нормализуется на масштаб линии, поэтому
    результат — метрика в пикселях.
    """
    p1 = undistort_pixels(c1, np.asarray(pixels1, dtype=float).reshape(-1, 2))
    p2 = undistort_pixels(c2, np.asarray(pixels2, dtype=float).reshape(-1, 2))
    F = fundamental_matrix(c1, c2)
    lines = np.c_[p1, np.ones(p1.shape[0])] @ F.T  # (N1, 3) эпиполярные линии по пикселям p1
    norm = np.hypot(lines[:, 0], lines[:, 1])
    norm[norm < 1e-12] = np.inf
    pts2 = np.c_[p2, np.ones(p2.shape[0])]  # (N2, 3)
    return np.abs(lines @ pts2.T) / norm[:, None]  # (N1, N2)


def epipolar_distance(
    c1: CameraCalibration, c2: CameraCalibration, pixel1: np.ndarray, pixel2: np.ndarray
) -> float:
    return float(
        epipolar_distance_batch(
            c1, c2, np.asarray(pixel1).reshape(1, 2), np.asarray(pixel2).reshape(1, 2)
        )[0]
    )


def baseline_length(c1: CameraCalibration, c2: CameraCalibration) -> float:
    """Расстояние между центрами проекции камер в мировых единицах."""
    o1 = -np.asarray(c1.R, dtype=float).T @ np.asarray(c1.t, dtype=float).reshape(3)
    o2 = -np.asarray(c2.R, dtype=float).T @ np.asarray(c2.t, dtype=float).reshape(3)
    return float(np.linalg.norm(o1 - o2))
