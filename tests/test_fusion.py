"""Слияние видов: сопоставление, триангуляция, трекинг (2 и 3 камеры, 1 объект)."""

import unittest

import numpy as np

from multiview3d.camera import project_point
from multiview3d.fusion import FusionConfig, MultiViewTracker, build_correspondences
from multiview3d.sim import make_sim_setup
from multiview3d.types import Detection2D


def oracle_dets(ds, t):
    """Идеальные детекции всех камер в момент t."""
    dets = []
    for cam in ds.cameras:
        pxs = [project_point(cam, np.asarray(p(t))) for p in ds.paths]
        dets.append([
            Detection2D(camera=cam.name, timestamp=t, center=np.array(px), score=1.0)
            for px in pxs
        ])
    return dets


class TestFusion(unittest.TestCase):
    def test_correspondences_two_objects(self):
        ds = make_sim_setup(num_cameras=3, num_objects=2, fps=25, duration_s=0.5, seed=7)
        dets = oracle_dets(ds, ds.frame_times[0])
        cfg = FusionConfig()
        groups, rejected = build_correspondences(ds.cameras, dets, cfg)
        self.assertEqual(len(groups), 2)
        self.assertEqual(rejected, [])
        for g in groups:
            self.assertEqual(len(g), 3)

    def test_triangulate_one_object_two_cameras(self):
        ds = make_sim_setup(num_cameras=2, num_objects=1, fps=25, duration_s=0.4, seed=11)
        t = ds.frame_times[0]
        dets = oracle_dets(ds, t)
        tr = MultiViewTracker([c.name for c in ds.cameras], ds.cameras, FusionConfig())
        res = tr.step(t, dets)
        self.assertEqual(len(res.candidates), 1)
        err_mm = np.linalg.norm(res.candidates[0].position - np.asarray(ds.paths[0](t))) * 1000
        self.assertLess(err_mm, 10.0)

    def test_tracking_stable_30_frames(self):
        """50 кадров синтетики (3 камеры, 2 объекта): ровно 2 трека, без пересозданий."""
        ds = make_sim_setup(num_cameras=3, num_objects=2, fps=25, duration_s=2.0, seed=7)
        cfg = FusionConfig(workspace_bounds=(-2, -2, 0, 2, 2, 3))
        tr = MultiViewTracker([c.name for c in ds.cameras], ds.cameras, cfg)
        next_id = 0
        history: dict[int, int] = {}
        for t in ds.frame_times:
            res = tr.step(t, oracle_dets(ds, t))
            for trk in res.tracks:
                history[trk.track_id] = len(trk.observations)
            next_id = max(next_id, tr._next_id - 1)
        self.assertEqual(next_id, 2)
        self.assertEqual(len(history), 2)
        self.assertGreater(min(history.values()), 30)


if __name__ == "__main__":
    unittest.main()
