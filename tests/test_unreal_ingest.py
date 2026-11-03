"""Тесты ingestion датасета «Пеленга» (§7.1 брифа) и геометрии камер.

Проверяется:
1. парсинг camera CSV (CP1251, ';', строки метаданных, номера кадров);
2. битые ячейки GT («авг.13», «июл.25») не превращаются в числа молча;
3. sidecar-репары помечены ``verified_by_organizer=false`` и не меняют файл;
4. единицы мира и трансляции совпадают (метры после x0.01);
5. конверсия осей UE -> OpenCV на известных точках;
6. статичность камер 1–4;
7. непрерывность yaw камеры 5 (нет искусственного скачка 360°);
8. выравнивание кадров: таймкод = frame / fps (PTS контейнером не отдаётся);
9. общая область кадров 0..1000 и отсутствие off-by-one на лишней строке 1001.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from multiview3d.camera import project_point
from multiview3d.truck_dataset import TruckDataset
from multiview3d.unreal_camera import (
    build_extrinsics,
    build_intrinsics,
    camera_from_row,
    load_ground_truth,
    parse_camera_csv,
    parse_track_csv,
    propose_repairs,
    unreal_to_right,
    UnrealRow,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "huniashahatona/zadania"
CAM5_CSV = DATA / "xyzpowcameta/object_transform_CineCameraActor5.csv"
CAM1_CSV = DATA / "xyzpowcameta/object_transform_CineCameraActor1.csv"
GT_CSV = DATA / "xyzpowcameta/object_transform_SM_AA_track.csv"
REPAIRS = ROOT / "datasets/truck/ground_truth_repairs.json"

CAMERA_HEADER = (
    "Frame;Loc_X;Loc_Y;Loc_Z;Rot_Roll;Rot_Pitch;Rot_Yaw;;;Focal length;20\n"
)
CAMERA_ROWS = (
    "0;100.0;200.0;300.0;0.0;0.0;-90.0;;;матрица;23,76*13,365мм\n"
    "1;100.0;200.0;300.0;0.0;0.0;-90.0;;;;\n"
    "2;100.0;200.0;300.0;0.0;0.0;-90.0;;;;\n"
)


def write_cp1251(path: Path, text: str) -> Path:
    path.write_bytes(text.encode("cp1251"))
    return path


class TestCameraCsvParse(unittest.TestCase):
    """§7.1 п.1: CP1251, ';', метаданные, номера кадров."""

    def test_parse_synthetic(self):
        with tempfile.TemporaryDirectory() as td:
            p = write_cp1251(Path(td) / "cam.csv", CAMERA_HEADER + CAMERA_ROWS)
            parsed = parse_camera_csv(p)
        self.assertEqual([r.frame for r in parsed.rows], [0, 1, 2])
        self.assertAlmostEqual(parsed.focal_mm, 20.0)
        self.assertEqual(parsed.sensor_mm, (23.76, 13.365))
        self.assertEqual(parsed.rows[0].loc, (100.0, 200.0, 300.0))
        self.assertEqual(parsed.rows[0].rot, (0.0, 0.0, -90.0))
        self.assertEqual(parsed.invalid_cells, [])

    def test_parse_real_camera5(self):
        parsed = parse_camera_csv(CAM5_CSV)
        self.assertGreater(len(parsed.rows), 1000)
        self.assertIsNotNone(parsed.focal_mm)
        self.assertIsNotNone(parsed.sensor_mm)
        self.assertEqual(parsed.rows[0].frame, 0)
        self.assertEqual(parsed.invalid_cells, [])

    def test_metadata_rows_not_frames(self):
        parsed = parse_camera_csv(CAM1_CSV)
        frames = [r.frame for r in parsed.rows]
        self.assertEqual(frames, sorted(frames))
        self.assertNotIn(-1, frames)


class TestGtInvalidCells(unittest.TestCase):
    """§7.1 п.2: битые ячейки обнаруживаются, а не конвертируются молча."""

    def test_invalid_cells_detected(self):
        parsed = parse_track_csv(GT_CSV)
        cells = {(c["frame"], c["column"], c["raw_value"]) for c in parsed.invalid_cells}
        self.assertIn((402, "Loc_X", "авг.13"), cells)
        self.assertIn((691, "Loc_Y", "июл.25"), cells)

    def test_frames_excluded_from_gt(self):
        gt = load_ground_truth(GT_CSV, repairs_path=REPAIRS)
        frames = set(int(f) for f in gt.frames)
        self.assertNotIn(402, frames)
        self.assertNotIn(691, frames)
        self.assertEqual(gt.repaired_frames, [])
        self.assertEqual(len(gt.invalid_cells), 2)
        self.assertEqual(len(frames), 1000)

    def test_strict_raises(self):
        with self.assertRaises(ValueError):
            load_ground_truth(GT_CSV, repairs_path=None, strict=True)

    def test_propose_repairs_hypothesis(self):
        parsed = parse_track_csv(GT_CSV)
        props = propose_repairs(parsed.invalid_cells)
        self.assertEqual(len(props), 2)
        by_frame = {p["frame"]: p for p in props}
        self.assertAlmostEqual(by_frame[402]["proposed_value"], 8.13)
        self.assertAlmostEqual(by_frame[691]["proposed_value"], 7.25)
        self.assertTrue(all(p["verified_by_organizer"] is False for p in props))


class TestGtOverrideAudit(unittest.TestCase):
    """§7.1 п.3: sidecar-репар явно маркирован и не меняет исходный файл."""

    def test_sidecar_does_not_modify_source(self):
        gt_text = ("Frame;Loc_X;Loc_Y;Loc_Z\n"
                   "0;0.0;0.0;100.0\n"
                   "1;авг.13;50.0;100.0\n"
                   "2;200.0;60.0;100.0\n")
        with tempfile.TemporaryDirectory() as td:
            gt = write_cp1251(Path(td) / "gt.csv", gt_text)
            before = gt.read_bytes()
            repairs = Path(td) / "repairs.json"
            repairs.write_text(json.dumps({
                "accept_unverified": False,
                "repairs": [{"frame": 1, "column": "Loc_X", "proposed_value": 8.13,
                             "source_raw_value": "авг.13",
                             "reason": "тест", "verified_by_organizer": False}],
            }), encoding="utf-8")
            strict = load_ground_truth(gt, repairs_path=repairs)
            accepted = load_ground_truth(gt, repairs_path=repairs, accept_unverified=True)
            with self.assertRaises(ValueError):
                load_ground_truth(gt, repairs_path=repairs, strict=True)
            self.assertEqual(gt.read_bytes(), before)

        self.assertNotIn(1, set(int(f) for f in strict.frames))
        self.assertEqual(strict.repaired_frames, [])
        self.assertIn(1, set(int(f) for f in accepted.frames))
        self.assertEqual(accepted.repaired_frames, [1])
        idx = list(accepted.frames).index(1)
        self.assertAlmostEqual(accepted.positions_m[idx][0], 0.0813, places=6)


class TestCameraUnits(unittest.TestCase):
    """§7.1 п.4: мир и трансляция в одних единицах (метры)."""

    def test_world_and_translation_meters(self):
        parsed = parse_camera_csv(CAM1_CSV)
        row = parsed.rows[0]
        cam = camera_from_row(row, "cam1", parsed.focal_mm, parsed.sensor_mm, 1920, 1080)
        center = -cam.R.T @ cam.t
        expected = unreal_to_right(*row.loc)
        np.testing.assert_allclose(center, expected, atol=1e-9)
        # 1 см UE = 0.01 м: центр камеры в метрах, а не в сантиметрах
        np.testing.assert_allclose(center * 100.0, np.array(row.loc) * np.array([1, -1, 1]))
        self.assertLess(np.linalg.norm(center), 10_000.0)

    def test_intrinsics_scale_with_resolution(self):
        K = build_intrinsics(20.0, (23.76, 13.365), 1920, 1080)
        self.assertAlmostEqual(K[0, 0], 20.0 * 1920 / 23.76)
        self.assertAlmostEqual(K[1, 1], 20.0 * 1080 / 13.365)
        self.assertAlmostEqual(K[0, 2], 960.0)
        self.assertAlmostEqual(K[1, 2], 540.0)
        K2 = build_intrinsics(20.0, (23.76, 13.365), 960, 540)
        self.assertAlmostEqual(K2[0, 0], K[0, 0] / 2.0)


class TestUnrealToOpenCvAxes(unittest.TestCase):
    """§7.1 п.5: известные точки/направления переводятся правильно."""

    def _cam(self):
        parsed = parse_camera_csv(CAM1_CSV)
        # камера в начале координат, без поворота: X вперёд, Y вправо, Z вверх
        row = UnrealRow(0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        return camera_from_row(row, "test", parsed.focal_mm, parsed.sensor_mm, 1920, 1080)

    def test_axis_directions(self):
        cam = self._cam()
        center = project_point(cam, unreal_to_right(1000.0, 0.0, 0.0))
        right = project_point(cam, unreal_to_right(1000.0, 1000.0, 0.0))
        up = project_point(cam, unreal_to_right(1000.0, 0.0, 1000.0))
        np.testing.assert_allclose(center, [960.0, 540.0], atol=1e-6)
        self.assertGreater(right[0], 960.0)   # +Y UE (вправо) -> правее центра
        self.assertAlmostEqual(right[1], 540.0, places=6)
        self.assertLess(up[1], 540.0)         # +Z UE (вверх) -> выше центра
        self.assertAlmostEqual(up[0], 960.0, places=6)

    def test_behind_camera_raises(self):
        cam = self._cam()
        with self.assertRaises(ValueError):
            project_point(cam, unreal_to_right(-1000.0, 0.0, 0.0))

    def test_right_handed_frames(self):
        parsed = parse_camera_csv(CAM1_CSV)
        r, t, _ = build_extrinsics(parsed.rows[0])
        self.assertAlmostEqual(float(np.linalg.det(r)), 1.0, places=9)
        self.assertEqual(t.shape, (3,))


class TestStaticCameraPose(unittest.TestCase):
    """§7.1 п.6: камеры 1–4 статичны по всей записи."""

    @classmethod
    def setUpClass(cls):
        cls.ds = TruckDataset.discover()

    def test_cameras_1_4_static(self):
        for name in ("cam1", "cam2", "cam3", "cam4"):
            self.assertTrue(self.ds.provider.is_static(name), name)
            cams = [self.ds.provider.calibration(name, f) for f in (0, 500, 1000)]
            self.assertTrue(all(c is cams[0] for c in cams), name)
            for cam in cams[1:]:
                np.testing.assert_array_equal(cam.R, cams[0].R)
                np.testing.assert_array_equal(cam.t, cams[0].t)

    def test_cam5_moves(self):
        self.assertFalse(self.ds.provider.is_static("cam5"))
        a = self.ds.provider.calibration("cam5", 0)
        b = self.ds.provider.calibration("cam5", 1000)
        self.assertGreater(np.abs(a.R - b.R).max(), 1e-3)


class TestCamera5YawUnwrap(unittest.TestCase):
    """§7.1 п.7: нет скачка 360° — матрица поворота непрерывна."""

    def test_rotation_continuity(self):
        parsed = parse_camera_csv(CAM5_CSV)
        frames = [r.frame for r in parsed.rows]
        self.assertEqual(frames, list(range(len(frames))))
        rots = [camera_from_row(r, "cam5", parsed.focal_mm, parsed.sensor_mm, 1920, 1080)
                for r in parsed.rows]
        angles = []
        for a, b in zip(rots[:-1], rots[1:]):
            cos = (np.trace(a.R.T @ b.R) - 1.0) / 2.0
            angles.append(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
        angles = np.asarray(angles)
        self.assertLess(float(angles.max()), 5.0,
                        f"разрыв в позе: max {angles.max():.2f}°/кадр")
        self.assertGreater(float(angles.sum()), 1.0, "камера 5 должна вращаться")

    def test_yaw_is_wrapped_in_csv(self):
        """В CSV yaw cam5 равен ±180° (wrap), поза при этом непрерывна."""
        parsed = parse_camera_csv(CAM5_CSV)
        yaws = np.array([r.rot[2] for r in parsed.rows])
        self.assertLessEqual(float(np.abs(yaws).max()), 180.0)


class TestVideoFrameAlignment(unittest.TestCase):
    """§7.1 п.8: выравнивание по номеру кадра; PTS контейнером не отдаётся."""

    @classmethod
    def setUpClass(cls):
        cls.ds = TruckDataset.discover()

    def test_timestamp_is_frame_over_fps(self):
        cam = self.ds.cameras[0]
        from multiview3d.truck_dataset import TruckVideoSource
        src = TruckVideoSource(cam, start=10, end=12, fps=self.ds.fps)
        try:
            stamps = []
            while True:
                item = src.read()
                if item is None:
                    break
                stamps.append(item[0])
            self.assertEqual(len(stamps), 3)
            for got, want in zip(stamps, (10 / 30, 11 / 30, 12 / 30)):
                self.assertAlmostEqual(got, want, places=9)
        finally:
            src.close()

    def test_sources_cover_same_frames(self):
        """Все пять видео отдают одинаковое число кадров при одном диапазоне."""
        counts = {}
        for cam in self.ds.cameras:
            from multiview3d.truck_dataset import TruckVideoSource
            src = TruckVideoSource(cam, start=0, end=4, fps=self.ds.fps)
            try:
                n = 0
                while src.read() is not None:
                    n += 1
            finally:
                src.close()
            counts[cam.name] = n
        self.assertEqual(set(counts.values()), {5}, counts)

    def test_pts_not_available_documented(self):
        """OpenCV не даёт PTS: выравнивание по frame index — рабочее допущение."""
        import cv2
        cap = cv2.VideoCapture(str(self.ds.cameras[0].video))
        try:
            self.assertFalse(hasattr(cap, "get_pts"))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
        finally:
            cap.release()
        self.assertAlmostEqual(fps, 30.0, places=3)


class TestFrameCountMismatch(unittest.TestCase):
    """§7.1 п.9: общая область 0..1000, лишняя строка CSV 1001 не даёт off-by-one."""

    @classmethod
    def setUpClass(cls):
        cls.ds = TruckDataset.discover()
        cls.report = cls.ds.mismatch_report()

    def test_common_range(self):
        self.assertEqual(self.report["common_range"], [0, 1000])
        self.assertEqual(self.report["common_frames"], 1001)

    def test_extra_csv_rows(self):
        for name, info in self.report["per_camera"].items():
            self.assertEqual(info["video_frames"], 1001, name)
            self.assertEqual(info["video_last_index"], 1000, name)
            self.assertEqual(info["csv_frames"], 1002, name)
            self.assertEqual(info["csv_rows_without_video"], 1, name)
            self.assertEqual(info["csv_last"], 1001, name)

    def test_open_sources_respect_range(self):
        from multiview3d.truck_dataset import TruckVideoSource
        src = TruckVideoSource(self.ds.cameras[0], start=1000, end=1000, fps=self.ds.fps)
        try:
            item = src.read()
            self.assertIsNotNone(item)
            self.assertAlmostEqual(item[0], 1000 / 30, places=9)
            self.assertIsNone(src.read())
        finally:
            src.close()


if __name__ == "__main__":
    unittest.main()
