"""Тесты отрисовки пути в пульте: сглаживание, разрывы, выбор трека."""

import unittest
from types import SimpleNamespace

import numpy as np

from apps.dashboard import Tracker, _JUMP, _smooth_run
from multiview3d.types import MEASURED, PREDICTED_ONLY, Track, TrackPoint


def _point(t: float, x: float, y: float, status: str = MEASURED) -> TrackPoint:
    return TrackPoint(timestamp=t, position=np.array([x, y, 0.0]),
                      velocity=np.zeros(3), status=status)


class TestSmoothing(unittest.TestCase):
    def test_straight_line_is_unchanged(self):
        """Равномерное движение сглаживание не сдвигает и не замедляет."""
        t = np.arange(40, dtype=float)
        arr = np.stack([t, 2.0 * t + 5.0, np.zeros_like(t)], axis=1)
        out = _smooth_run(arr, 3)
        np.testing.assert_allclose(out, arr, atol=1e-9)

    def test_noise_is_reduced(self):
        t = np.arange(60, dtype=float)
        clean = np.stack([t, np.zeros_like(t), np.zeros_like(t)], axis=1)
        noisy = clean.copy()
        noisy[:, 1] = 2.0 * (-1.0) ** t          # детерминированный «пилообразный» шум
        out = _smooth_run(noisy, 3)
        self.assertLess(out[5:-5, 1].std(), noisy[5:-5, 1].std() * 0.5)

    def test_short_run_and_zero_window(self):
        arr = np.arange(6, dtype=float).reshape(2, 3)
        np.testing.assert_allclose(_smooth_run(arr, 0), arr)
        np.testing.assert_allclose(_smooth_run(arr, 3), arr)


class TestTrailBreaks(unittest.TestCase):
    def test_none_splits_runs(self):
        trail = [(np.zeros(3), MEASURED), (np.ones(3), MEASURED), None,
                 (np.full(3, 2.0), PREDICTED_ONLY)]
        runs = Tracker._path_runs(trail)
        self.assertEqual(len(runs), 2)
        self.assertEqual(runs[0][0], MEASURED)
        self.assertEqual(len(runs[0][1]), 2)
        self.assertEqual(runs[1][0], PREDICTED_ONLY)

    def test_jump_splits_runs_without_connector(self):
        trail = [(np.zeros(3), MEASURED), _JUMP, (np.full(3, 9.0), MEASURED)]
        self.assertEqual(len(Tracker._path_runs(trail)), 2)
        self.assertEqual(list(Tracker._gap_pairs(trail)), [])

    def test_gap_connects_only_time_gaps(self):
        a, b, c, d = (np.zeros(3), np.ones(3), np.full(3, 2.0), np.full(3, 3.0))
        trail = [(a, MEASURED), None, (b, MEASURED), _JUMP, (c, MEASURED),
                 None, (d, MEASURED)]
        pairs = list(Tracker._gap_pairs(trail))
        self.assertEqual(len(pairs), 2)
        np.testing.assert_allclose(pairs[0][0], a)
        np.testing.assert_allclose(pairs[0][1], b)
        np.testing.assert_allclose(pairs[1][1], d)


class TestTrackChoice(unittest.TestCase):
    def test_measured_beats_predicted(self):
        ts = 1.0
        old = Track(1, [_point(ts, 0.0, 0.0, PREDICTED_ONLY)])
        new = Track(2, [_point(ts, 10.0, 0.0, MEASURED)])
        result = SimpleNamespace(tracks=[old, new])
        self.assertIs(Tracker._pick_track(result), new)

    def test_newer_timestamp_wins(self):
        old = Track(1, [_point(2.0, 0.0, 0.0)])
        new = Track(2, [_point(3.0, 10.0, 0.0)])
        result = SimpleNamespace(tracks=[old, new])
        self.assertIs(Tracker._pick_track(result), new)

    def test_empty_result(self):
        self.assertIsNone(Tracker._pick_track(SimpleNamespace(tracks=[])))


if __name__ == "__main__":
    unittest.main()
