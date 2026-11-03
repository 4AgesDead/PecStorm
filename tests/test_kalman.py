"""Фильтр Калмана: сходимость на линейном движении, сглаживание RTS."""

import unittest

import numpy as np

from multiview3d.kalman import KalmanFilter, rts_smooth


class TestKalman(unittest.TestCase):
    def test_linear_convergence(self):
        """Линейное движение v=(1,0,0), шум измерений 1 см -> ошибка позиции ~11 мм,
        скорость ~0.04 м/с (замеры на seed=3, пороги с запасом)."""
        fps, n = 25.0, 60
        v = np.array([1.0, 0.0, 0.0])
        rng = np.random.default_rng(3)
        kf = KalmanFilter(np.array([0, 0, 0, 0, 0, 0], float),
                          np.diag([0.01 ** 2] * 3 + [4.0] * 3),
                          (0.001, 0.02), 0.01 ** 2)
        kf.t = 0.0
        for i in range(1, n + 1):
            t = i / fps
            kf.predict(t)
            z = v * t + rng.normal(0, 0.01, 3)
            kf.update(z)
        self.assertLess(np.linalg.norm(kf.x[:3] - v * (n / fps)), 0.02)
        self.assertLess(np.linalg.norm(kf.x[3:] - v), 0.06)

    def test_mahalanobis_gate(self):
        kf = KalmanFilter(np.array([0, 0, 0, 0, 0, 0], float),
                          np.eye(6) * 0.01, (0.01, 0.1), 0.01)
        self.assertLess(kf.mahalanobis(np.array([0.01, 0.0, 0.0])), 9.0)
        self.assertGreater(kf.mahalanobis(np.array([2.0, 0.0, 0.0])), 9.0)

    def test_rts_smoothing_reduces_noise(self):
        """RTS-сглаживание уменьшает разброс относительно истинной прямой."""
        fps, n = 25.0, 60
        v = np.array([0.5, 0.2, 0.0])
        rng = np.random.default_rng(5)
        kf = KalmanFilter(np.array([0, 0, 0, 0, 0, 0], float),
                          np.diag([0.01 ** 2] * 3 + [4.0] * 3),
                          (0.001, 0.5), 0.01 ** 2)
        kf.t = 0.0
        times, states, covs = [], [], []
        filt_err = []
        for i in range(1, n + 1):
            t = i / fps
            kf.predict(t)
            z = v * t + rng.normal(0, 0.01, 3)
            kf.update(z)
            times.append(t)
            states.append(kf.x.copy())
            covs.append(kf.P.copy())
            filt_err.append(np.linalg.norm(kf.x[:3] - v * t))
        sm = rts_smooth(times, np.array(states), np.array(covs), 0.001, 0.5)
        sm_err = [np.linalg.norm(s.position - v * s.timestamp) for s in sm]
        self.assertLess(float(np.mean(sm_err)), float(np.mean(filt_err)))


if __name__ == "__main__":
    unittest.main()
