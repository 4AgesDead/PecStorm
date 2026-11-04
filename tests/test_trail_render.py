"""Тесты общей отрисовки пути: история, прогноз, окно карты."""

import unittest

import numpy as np

from multiview3d.trail_render import (JUMP, build_trail, forecast, map_window,
                                      mean_velocity, render_map_panel)


def _straight(n=30, dt=1 / 30, speed=15.0):
    return [(i * dt, np.array([i * speed * dt, 0.0, 0.0]), "MEASURED")
            for i in range(n)]


class TestBuildTrail(unittest.TestCase):
    def test_straight_line_keeps_length_and_tip(self):
        hist = _straight(30)
        trail = build_trail(hist, 30, 5)
        self.assertEqual(len(trail), 30)
        self.assertTrue(all(isinstance(item, tuple) for item in trail))
        np.testing.assert_allclose(trail[-1][0], hist[-1][1], atol=1e-6)

    def test_jump_becomes_gap_not_jump_sentinel(self):
        hist = _straight(10)
        hist.append((10 / 30, np.array([100.0, 0.0, 0.0]), "MEASURED"))
        trail = build_trail(hist, 20, 5)
        self.assertIn(None, trail)
        self.assertFalse(any(item is JUMP for item in trail))

    def test_zero_smoothing_returns_raw_points(self):
        hist = _straight(10)
        trail = build_trail(hist, 10, 0)
        for (p, _), (_, q, _) in zip(trail, hist):
            np.testing.assert_allclose(p, q)


class TestForecast(unittest.TestCase):
    def test_direction_is_forward(self):
        hist = _straight(20)
        fc = forecast(hist, 3.0, 3.0)
        self.assertIsNotNone(fc)
        self.assertGreater(fc[-1][0], hist[-1][1][0])
        self.assertEqual(len(fc), 16)

    def test_too_short_history_has_no_velocity(self):
        self.assertIsNone(mean_velocity(_straight(3)))
        self.assertIsNone(forecast(_straight(3), 3.0, 3.0))


class TestMapPanel(unittest.TestCase):
    def test_gt_overlay_is_drawn(self):
        trail = [(np.array([0.0, 0.0, 0.0]), "MEASURED"),
                 (np.array([10.0, 0.0, 0.0]), "MEASURED")]
        gt = [np.array([0.0, 0.0, 0.0]), np.array([10.0, 0.0, 0.0])]
        canvas, _ = render_map_panel(
            trail, None, np.array([10.0, 0.0, 0.0]), "MEASURED", 1.0, 10,
            gt_points=gt, gt_position=np.array([10.0, 0.0, 0.0]))
        self.assertEqual(canvas.shape, (640, 900, 3))
        # жёлтый BGR (0, 220, 255): синий < 80, зелёный > 150, красный > 200
        yellow = ((canvas[:, :, 0] < 80) & (canvas[:, :, 1] > 150)
                  & (canvas[:, :, 2] > 200))
        self.assertTrue(yellow.any())


class TestMapWindow(unittest.TestCase):
    def test_expands_immediately_and_shrinks_smoothly(self):
        small = [np.array([-10.0, -10.0]), np.array([10.0, 10.0])]
        view = map_window(small, None)
        span0 = view[0]
        self.assertGreaterEqual(span0, 60.0)

        wide = [np.array([-100.0, -20.0]), np.array([100.0, 20.0])]
        span1 = map_window(wide, view)[0]
        self.assertGreater(span1, span0)   # расширение сразу

        span2 = map_window(small, (span1, 0.0, 0.0))[0]
        self.assertLess(span2, span1)      # сжатие постепенное
        self.assertGreaterEqual(span2, 60.0)


if __name__ == "__main__":
    unittest.main()
