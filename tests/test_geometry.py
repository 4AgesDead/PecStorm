"""Геометрия проекции/триангуляции/эпиполярной согласованности."""

import unittest

import numpy as np

from multiview3d.camera import (
    back_project_ray,
    epipolar_distance,
    fundamental_matrix,
    project_point,
    project_points,
    refine_point,
    reprojection_error,
    triangulate_rays,
    undistort_pixels,
)
from multiview3d.calibration import correspondence_epipolar_errors
from multiview3d.sim import cam_center, make_sim_setup


class TestGeometry(unittest.TestCase):
    def setUp(self) -> None:
        self.ds = make_sim_setup(num_cameras=3, num_objects=2, fps=25, duration_s=0.5)
        self.X = self.ds.truth_at(0.0)

    def test_projection_roundtrip(self):
        """Проекция/обратный луч: луч проходит через точку; репроекция ~0."""
        cam = self.ds.cameras[0]
        px = project_point(cam, self.X[0])
        ray = back_project_ray(cam, px)
        C = cam_center(cam)
        t = np.dot(np.asarray(self.X[0]) - C, ray)
        X_back = C + t * ray
        self.assertLess(np.linalg.norm(X_back - self.X[0]), 1e-6)
        self.assertLess(reprojection_error(cam, self.X[0], px), 1e-6)

    def test_behind_camera_nan(self):
        """Точка за камерой -> NaN в project_points."""
        cam = self.ds.cameras[0]
        behind = cam_center(cam) - (self.ds.cameras[0].R.T @ np.array([0.0, 0.0, 1.0])) * 2.0
        px = project_points(cam, [behind])
        self.assertTrue(np.isnan(px).all())

    def test_triangulate_exact(self):
        """DLT по лучам трёх камер восстанавливает истинную точку (< 0.1 мм)."""
        cams = self.ds.cameras
        pixels = [project_point(c, self.X[0]) for c in cams]
        rays = [back_project_ray(c, p) for c, p in zip(cams, pixels)]
        X = triangulate_rays(rays, [cam_center(c) for c in cams])
        self.assertLess(np.linalg.norm(X - self.X[0]) * 1000.0, 0.1)

    def test_refine_point(self):
        cams = self.ds.cameras
        pixels = [project_point(c, self.X[1]) for c in cams]
        X = np.asarray(self.X[1]) + np.array([0.05, -0.03, 0.04])
        Xr = refine_point(cams, pixels, X)
        self.assertLess(np.linalg.norm(Xr - self.X[1]) * 1000.0, 5.0)

    def test_epipolar_true_correspondence(self):
        """Истинные соответствия: эпиполярное расстояние ~0, ложные — большие."""
        c1, c2 = self.ds.cameras[0], self.ds.cameras[1]
        p1 = project_point(c1, self.X[0])
        p2 = project_point(c2, self.X[0])
        self.assertLess(epipolar_distance(c1, c2, p1, p2), 0.05)
        q2 = project_point(c2, self.X[1])
        self.assertGreater(epipolar_distance(c1, c2, p1, q2), 20.0)

    def test_epipolar_matrix_symmetric(self):
        c1, c2 = self.ds.cameras[0], self.ds.cameras[1]
        p1 = np.array([project_point(c1, x) for x in self.X])
        p2 = np.array([project_point(c2, x) for x in self.X])
        m = correspondence_epipolar_errors(c1, c2, p1, p2)
        self.assertEqual(m.shape, (2, 2))
        self.assertLess(m[0, 0], 0.1)
        self.assertLess(m[1, 1], 0.1)
        self.assertGreater(m[0, 1], 20.0)
        self.assertGreater(m[1, 0], 20.0)

    def test_fundamental_bilinear(self):
        c1, c2 = self.ds.cameras[0], self.ds.cameras[1]
        F = fundamental_matrix(c1, c2)
        u1 = undistort_pixels(c1, project_point(c1, self.X[0])[None])[0]
        u2 = undistort_pixels(c2, project_point(c2, self.X[0])[None])[0]
        self.assertLess(abs(float(np.r_[u2, 1.0] @ F @ np.r_[u1, 1.0])), 1e-6)


if __name__ == "__main__":
    unittest.main()
