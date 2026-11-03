"""Калибрация по синтетическому шахматному полигону (рендер точной моделью).

Кадры полигона рисуются идеальной пинхол-моделью и искажаются той же моделью
OpenCV, что и калибратор, поэтому результат калибрации сравним с «истинной» K.
"""

import unittest

import cv2
import numpy as np

from multiview3d.calibration import (
    board_object_points,
    calibrate_camera_from_images,
    load_calibration,
    reprojection_rms,
    save_calibration,
    solve_extrinsics_from_points,
)
from multiview3d.camera import project_points
from multiview3d.sim import look_at_rotation


def render_board_view(K, dist, R, t, pattern, square, width, height, border_sq=40):
    """Кадр шахматного полигона: честный пинхол + дисторсия через remap.

    Доска: (cols+3)x(rows+3) клеток — внутренний паттерн (cols+1)x(rows+1)
    клеток даёт cols x rows внутренних углов, внешняя рамка целиком белая
    (требование findChessboardCorners). Внутренние углы центрированы в 0,
    как в board_object_points(kind='chessboard').
    """
    cols, rows = pattern
    s = border_sq
    bw = (cols + 3) * s
    bh = (rows + 3) * s
    board = np.zeros((bh, bw, 3), np.uint8)
    for i in range(rows + 3):
        for j in range(cols + 3):
            ring = i == 0 or j == 0 or i == rows + 2 or j == cols + 2
            if ring or (i + j) % 2 == 1:
                board[i * s:(i + 1) * s, j * s:(j + 1) * s] = 255

    # board-пиксель -> плоскость полигона (центр в 0), плоскость -> изображение.
    sx = square / s
    S = np.array([[sx, 0, -(cols + 3) / 2 * square],
                  [0, sx, -(rows + 3) / 2 * square], [0, 0, 1]])
    H = K @ np.column_stack((R[:, 0], R[:, 1], t)) @ S
    ideal = cv2.warpPerspective(board, H, (width, height))
    if dist is not None and np.any(np.asarray(dist) != 0):
        # Карта «искажённый пиксель -> идеальный пиксель»: out(p) = ideal(undistort(p)).
        grid = np.stack(np.meshgrid(np.arange(width), np.arange(height)), axis=-1)
        und = cv2.undistortPoints(grid.reshape(-1, 1, 2).astype(np.float64),
                                  K, np.asarray(dist, np.float64), P=K)
        map_xy = und.reshape(height, width, 2).astype(np.float32)
        return cv2.remap(ideal, map_xy, None, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)
    return ideal


class TestCalibration(unittest.TestCase):
    def test_intrinsics_from_synthetic_board(self):
        cols, rows = 9, 6
        square = 0.03
        width, height = 640, 480
        K_true = np.array([[600.0, 0.0, width / 2], [0.0, 600.0, height / 2], [0, 0, 1]])
        dist_true = np.array([-0.05, 0.02, 0.001, 0.001, 0.0])

        # 8 ракурсов: разный радиус/высота и смещённая цель взгляда —
        # разнообразие нужен, иначе fy и cy плохо отделяются друг от друга.
        cfg = [(0.25, 0.35, (0.0, 0.0, 0.0)), (0.30, 0.55, (0.06, -0.05, 0.0)),
               (0.25, 0.45, (-0.06, 0.05, 0.0)), (0.35, 0.30, (0.05, 0.05, 0.0)),
               (0.28, 0.50, (-0.05, -0.06, 0.0)), (0.25, 0.40, (0.0, -0.06, 0.0)),
               (0.32, 0.60, (-0.04, 0.0, 0.0)), (0.27, 0.33, (0.04, 0.04, 0.0))]
        views = []
        for k, (r, z, tgt) in enumerate(cfg):
            origin = np.array([r * np.cos(k * np.pi / 4), r * np.sin(k * np.pi / 4), z])
            tg = np.array(tgt)
            R = look_at_rotation(origin, tg)
            t = -R @ origin
            views.append(render_board_view(K_true, dist_true, R, t, (cols, rows),
                                           square, width, height))

        cam, info = calibrate_camera_from_images("camX", views, (cols, rows),
                                                 square * 1000, "chessboard")
        self.assertGreaterEqual(info["views_used"], 5)
        self.assertLess(info["reprojection_rms_px"], 2.5)
        self.assertAlmostEqual(cam.K[0, 0] / K_true[0, 0], 1.0, delta=0.08)
        self.assertAlmostEqual(cam.K[1, 1] / K_true[1, 1], 1.0, delta=0.08)
        self.assertLess(abs(cam.K[0, 2] - K_true[0, 2]), 20)
        self.assertLess(abs(cam.K[1, 2] - K_true[1, 2]), 20)

        # Для каждого ракурса решаем внешние параметры (solvePnP по восстановленным
        # внутренним) и проверяем репроецию в исходные пиксели.
        obj = board_object_points((cols, rows), square)
        rms_max, n_views = 0.0, 0
        for v in views:
            gray = cv2.cvtColor(v, cv2.COLOR_BGR2GRAY)
            ok, corners = cv2.findChessboardCorners(gray, (cols, rows),
                                                    cv2.CALIB_CB_ADAPTIVE_THRESH)
            if not ok:
                continue
            pose = solve_extrinsics_from_points(cam, obj, corners.reshape(-1, 2))
            rms_max = max(rms_max, reprojection_rms(pose, obj, corners.reshape(-1, 2)))
            n_views += 1
        self.assertGreaterEqual(n_views, 4)
        self.assertLess(rms_max, 2.5)

    def test_save_load_roundtrip(self):
        import tempfile
        from pathlib import Path
        from multiview3d.types import CameraCalibration
        cam = CameraCalibration("c1", np.eye(3) * 2, np.zeros(5), np.eye(3),
                                np.array([0.0, 0.0, 1.0]), width=8, height=6, unit=1.0)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "calib.json"
            save_calibration(p, [cam], meta={"m": 1})
            cams, meta = load_calibration(p)
            self.assertEqual(len(cams), 1)
            self.assertEqual(cams[0].name, "c1")
            self.assertEqual(meta.get("m"), 1)
            np.testing.assert_allclose(cams[0].t, cam.t)

    def test_unit_must_be_number(self):
        from multiview3d.types import CameraCalibration
        with self.assertRaises(ValueError):
            CameraCalibration("c", np.eye(3), np.zeros(5), np.eye(3), np.zeros(3), unit="мм")


if __name__ == "__main__":
    unittest.main()
