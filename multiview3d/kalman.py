"""Фильтр Калмана с моделью постоянной скорости и сглаживание RTS."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _transition(dt: float) -> np.ndarray:
    """Матрица перехода для модели постоянной скорости (x, y, z, vx, vy, vz)."""
    f = np.eye(6)
    f[0, 3] = dt
    f[1, 4] = dt
    f[2, 5] = dt
    return f


def _process_cov(dt: float, pos_var: float, vel_var: float) -> np.ndarray:
    """Диагональная ковариация процессов: шум положения и шума скорости за шаг dt."""
    q = np.zeros(6)
    q[0:3] = pos_var * max(dt, 1e-6)
    q[3:6] = vel_var * max(dt, 1e-6)
    return np.diag(q)


class KalmanFilter:
    """Линейный фильтр Калмана 6D (положение + скорость), измерения — 3D-позиция.

    Параметры:
        x0: начальное состояние [x, y, z, vx, vy, vz].
        P0: начальная ковариация состояния (6x6).
        Q_diag: [шум_положения_на_сек, шум_скорости_на_сек] (дисперсии на 1 секунду).
        measurement_noise: дисперсия шума измерения позиции (в единицах calibration.unit^2).
    """

    def __init__(
        self,
        x0: np.ndarray,
        P0: np.ndarray,
        Q_diag: list[float] | tuple[float, float],
        measurement_noise: float,
    ) -> None:
        self.x = np.asarray(x0, dtype=float).reshape(6)
        self.P = np.asarray(P0, dtype=float).reshape(6, 6)
        self.pos_var = float(Q_diag[0])
        self.vel_var = float(Q_diag[1])
        self.R = float(measurement_noise)
        self.t = 0.0
        self._H = np.eye(3, 6)
        self._S_inv = np.linalg.inv(self.P[:3, :3] + np.eye(3) * self.R)

    def predict(self, t: float) -> np.ndarray:
        """Перенос состояния на время t (секунды), dt = t - self.t."""
        dt = max(float(t) - self.t, 0.0)
        f = _transition(dt)
        self.x = f @ self.x
        self.P = f @ self.P @ f.T + _process_cov(dt, self.pos_var, self.vel_var)
        self.t = float(t)
        self._update_S_inv()
        return self.x

    def innovation_covariance(self) -> np.ndarray:
        """S = H P H^T + R — ковариация инновации в пространстве положений."""
        return self.P[:3, :3] + np.eye(3) * self.R

    def _update_S_inv(self) -> None:
        self._S_inv = np.linalg.inv(self.innovation_covariance())

    def mahalanobis(self, z: np.ndarray) -> float:
        """Махаланобисово расстояние предсказанного состояния до измерения z."""
        diff = np.asarray(z, dtype=float).reshape(3) - self.x[:3]
        return float(diff @ self._S_inv @ diff)

    def update(self, z: np.ndarray) -> np.ndarray:
        """Коррекция по измерению 3D-позиции z."""
        z = np.asarray(z, dtype=float).reshape(3)
        h = self._H
        s = self.innovation_covariance()
        k = self.P @ h.T @ np.linalg.inv(s)
        innovation = z - h @ self.x
        self.x = self.x + k @ innovation
        i_kh = np.eye(6) - k @ h
        self.P = i_kh @ self.P @ i_kh.T + k @ np.eye(3) * self.R @ k.T
        self._update_S_inv()
        return self.x


@dataclass
class SmoothedState:
    timestamp: float
    position: np.ndarray
    velocity: np.ndarray


def rts_smooth(times: list[float], states: np.ndarray, covariances: np.ndarray,
               pos_var: float, vel_var: float) -> list[SmoothedState]:
    """Офлайн-сглаживание Rauch-Tung-Striebel для последовательности фильтрованных состояний.

    states: (N, 6) — состояния после update(); covariances: (N, 6, 6) — P после update().
    Возвращает сглаженные позиции и скорости.
    """
    n = len(times)
    if n == 0:
        return []
    xs = [np.asarray(s, dtype=float).reshape(6) for s in states]
    ps = [np.asarray(p, dtype=float).reshape(6, 6) for p in covariances]
    # Проход вперёд: предсказания от последнего фильтра (x_pred[i], P_pred[i], F[i]).
    x_pred = [None] * n
    p_pred = [None] * n
    fs: list[np.ndarray] = []
    for i in range(1, n):
        dt = max(float(times[i]) - float(times[i - 1]), 1e-6)
        f = _transition(dt)
        fs.append(f)
        x_pred[i] = f @ xs[i - 1]
        p_pred[i] = f @ ps[i - 1] @ f.T + _process_cov(dt, pos_var, vel_var)
    xs_smooth = list(xs)
    ps_smooth = list(ps)
    # Проход назад.
    for i in range(n - 2, -1, -1):
        f = fs[i]
        p_pred_i = p_pred[i + 1]
        try:
            c = ps[i] @ f.T @ np.linalg.inv(p_pred_i)
        except np.linalg.LinAlgError:
            c = ps[i] @ f.T @ np.linalg.pinv(p_pred_i)
        xs_smooth[i] = xs[i] + c @ (xs_smooth[i + 1] - x_pred[i + 1])
        ps_smooth[i] = ps[i] + c @ (ps_smooth[i + 1] - p_pred_i) @ c.T
    return [
        SmoothedState(timestamp=float(times[i]), position=xs_smooth[i][:3].copy(),
                      velocity=xs_smooth[i][3:].copy())
        for i in range(n)
    ]
