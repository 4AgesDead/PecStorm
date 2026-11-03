"""Конвейер и метрики: сквозной офлайн-прогон синтетики, экспорт CSV/JSON."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from multiview3d.fusion import FusionConfig
from multiview3d.metrics import evaluate
from multiview3d.pipeline import MultiViewPipeline, PipelineConfig
from multiview3d.sim import OracleSimSource, make_sim_setup
from multiview3d.visualize import tracks_from_csv, tracks_to_csv, truth_from_csv, truth_to_csv


class TestPipeline(unittest.TestCase):
    def test_offline_end_to_end_accuracy(self):
        """Критерий приёмки: 3 камеры, 2 объекта, 2 с -> 2 трека, ошибка <= 30 мм."""
        ds = make_sim_setup(num_cameras=3, num_objects=2, fps=25, duration_s=2.0, seed=7)
        cfg = PipelineConfig(cameras=ds.cameras,
                             fusion=FusionConfig(workspace_bounds=(-2, -2, 0, 2, 2, 3)))
        pipe = MultiViewPipeline(ds.cameras, cfg)
        sources = [OracleSimSource(ds, i) for i in range(len(ds.cameras))]
        tracks = pipe.run_offline(sources)
        smoothed = pipe.smoothed_tracks()
        truth = [[np.asarray(p(t)) for t in ds.frame_times] for p in ds.paths]

        self.assertEqual(len(tracks), 2)
        rep = evaluate(smoothed, truth, ds.labels, fps=ds.fps, unit=1.0,
                       pipeline_stats=pipe.stats())
        self.assertEqual(len(rep.pairs), 2)
        self.assertEqual(rep.unmatched_truth, [])
        self.assertEqual(rep.id_switches, 0)
        self.assertLessEqual(rep.overall.mean_mm, 30.0)
        self.assertGreaterEqual(min(rep.completeness.values()), 0.8)

    def test_csv_json_roundtrip(self):
        ds = make_sim_setup(num_cameras=2, num_objects=1, fps=25, duration_s=0.4, seed=3)
        truth = [np.asarray([p(t) for t in ds.frame_times]) for p in ds.paths]
        from multiview3d.types import Track, TrackPoint
        track = Track(1, [TrackPoint(t, truth[0][i], np.zeros(3), ["a"], 0.5)
                          for i, t in enumerate(ds.frame_times)])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.csv"
            tracks_to_csv([track], p)
            back = tracks_from_csv(p)
            self.assertEqual(len(back), 1)
            self.assertEqual(len(back[0].points), len(ds.frame_times))
            np.testing.assert_allclose(back[0].points[0].position, truth[0][0], atol=1e-5)
            tp = Path(d) / "truth.csv"
            truth_to_csv([truth[0]], ["obj"], ds.frame_times, tp)
            truth_back, times = truth_from_csv(tp)
            self.assertEqual(len(times), len(ds.frame_times))
            np.testing.assert_allclose(times, ds.frame_times, atol=1e-6)
            np.testing.assert_allclose(truth_back["obj"], truth[0], atol=1e-5)


if __name__ == "__main__":
    unittest.main()
