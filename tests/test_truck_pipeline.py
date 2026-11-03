"""Тесты truck-workflow: проекция в видео, шум 2D, трекинг, пропуски видов,
источник с детектором, обмен детекциями, метрики с coverage, динамическая камера.

Синтетические проверки (§7.4/7.5 брифа) идут на геометрии ``make_sim_setup``:
там известна истинная 3D-точка, поэтому можно измерить ошибку, а не только
репроекционную невязку. Реальные данные используются там, где нужна именно
проверка ingestion/поз (проекция GT-пивота в кадры, вращение cam5).
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from multiview3d.camera import project_point
from multiview3d.capture import FrameSource
from multiview3d.fusion import (
    FusionConfig,
    MultiViewTracker,
    build_correspondences,
    canonical_view_key,
)
from multiview3d.metrics import evaluate_truck
from multiview3d.pipeline import PipelineConfig
from multiview3d.sim import make_sim_setup
from multiview3d.sources import DetectingFrameSource
from multiview3d.truck_dataset import TruckDataset
from multiview3d.truck_pipeline import (
    build_pipeline,
    detection_rows,
    load_detections,
    save_detections,
)
from multiview3d.types import (
    MEASURED,
    PREDICTED_ONLY,
    Detection2D,
    TrackPoint,
)
from multiview3d.visualize import plot_components_vs_time


class TestProjectionToVideo(unittest.TestCase):
    """§7.2: GT-пивот проецируется в кадр; проверка для всех камер и cam5 в повороте."""

    @classmethod
    def setUpClass(cls):
        cls.ds = TruckDataset.discover()
        cls.gt = cls.ds.ground_truth_by_frame()

    def test_pivot_projects_inside_frames(self):
        frames = [f for f in range(0, 1001, 100) if f in self.gt]
        inside_total = 0
        for f in frames:
            cams = self.ds.provider.calibrations_at(f, nearest=True)
            for name, cam in cams.items():
                try:
                    uv = project_point(cam, self.gt[f])
                except ValueError:
                    continue
                if np.all(np.isfinite(uv)) and 0 <= uv[0] < cam.width and 0 <= uv[1] < cam.height:
                    inside_total += 1
        per_frame = inside_total / len(frames)
        self.assertGreater(per_frame, 1.0,
                           "в среднем должно быть больше одного вида с пивотом в кадре")

    def test_camera5_rotating_projection_finite(self):
        f = 750
        cam = self.ds.provider.calibration("cam5", f)
        uv = project_point(cam, self.gt[f])
        self.assertTrue(np.all(np.isfinite(uv)))
        self.assertGreater(float(np.linalg.norm(uv)), 0.0)

    def test_projection_changes_with_camera_pose(self):
        a = project_point(self.ds.provider.calibration("cam5", 0), self.gt[0])
        b = project_point(self.ds.provider.calibration("cam5", 1000), self.gt[1000])
        self.assertGreater(float(np.linalg.norm(a - b)), 1.0)


class TestDetectionsExchange(unittest.TestCase):
    """Обмен 2D-детекциями между этапами: CSV без потерь значимых полей."""

    def test_roundtrip(self):
        import tempfile
        dets = {
            10: {"cam1": [Detection2D("cam1", 10 / 30, np.array([100.5, 200.25]),
                                      score=0.9, bbox=(10.0, 20.0, 190.0, 380.0), label="7")],
                 "cam2": [Detection2D("cam2", 10 / 30, np.array([50.0, 60.0]),
                                      score=0.5, bbox=None, label="7")]},
            20: {"cam3": [Detection2D("cam3", 20 / 30, np.array([7.0, 8.0]), score=0.3)]},
        }
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "dets.csv"
            n = save_detections(detection_rows(dets), p)
            back = load_detections(p)
        self.assertEqual(n, 3)
        self.assertEqual(sorted(back), [10, 20])
        d = back[10]["cam1"][0]
        np.testing.assert_allclose(d.center, [100.5, 200.25], atol=1e-3)
        self.assertAlmostEqual(d.score, 0.9, places=4)
        self.assertIsNotNone(d.bbox)
        self.assertAlmostEqual(d.bbox[3], 380.0, places=3)
        self.assertIsNone(back[10]["cam2"][0].bbox)
        self.assertEqual(back[20]["cam3"][0].camera, "cam3")


class _StubSource(FrameSource):
    """Источник из N кадров и последующего исчерпания."""

    def __init__(self, n: int = 2, fps: float = 30.0):
        self.name = "camX"
        self.n = n
        self.i = 0
        self.fps = fps
        self.closed = False

    def read(self):
        if self.i >= self.n:
            return None
        frame = np.zeros((48, 64, 3), np.uint8)
        stamp = self.i / self.fps
        self.i += 1
        return stamp, frame, None

    def close(self) -> None:
        self.closed = True


class _StubDetector:
    """Детектор: одна детекция в кадре и одна за его границей."""

    def __init__(self):
        self.calls = 0

    def detect(self, frame_bgr):
        self.calls += 1
        return [Detection2D("", 0.0, np.array([10.0, 20.0]), 0.9),
                Detection2D("", 0.0, np.array([10_000.0, 20.0]), 0.9)]


class TestDetectorSource(unittest.TestCase):
    """§P0.3: детектор реально запускается на кадрах источника."""

    def test_detector_applied_and_named(self):
        src = DetectingFrameSource(_StubSource(), _StubDetector(), name="cam7")
        item = src.read()
        self.assertIsNotNone(item)
        stamp, frame, dets = item
        self.assertEqual(len(dets), 1, "детекция вне кадра должна отбрасываться")
        self.assertEqual(dets[0].camera, "cam7")
        self.assertAlmostEqual(dets[0].timestamp, 0.0)
        self.assertIsInstance(frame, np.ndarray)

    def test_runs_once_per_frame(self):
        det = _StubDetector()
        src = DetectingFrameSource(_StubSource(n=3), det)
        for _ in range(3):
            self.assertIsNotNone(src.read())
        self.assertIsNone(src.read())
        self.assertEqual(det.calls, 3)
        self.assertEqual(src.frames_detected, 3)


class TestTriangulationNoise(unittest.TestCase):
    """§7.4: устойчивость триангуляции к шуму 2D-точек (истинная 3D-точка известна)."""

    def setUp(self):
        self.ds = make_sim_setup(num_cameras=3, num_objects=1, fps=25,
                                 duration_s=0.4, seed=11)
        self.t = self.ds.frame_times[0]
        self.truth = np.asarray(self.ds.paths[0](self.t), dtype=float)

    def _triangulate(self, noise_px: float, seed: int = 5) -> float:
        rng = np.random.default_rng(seed)
        cfg = FusionConfig(association_mode="single_target", max_reprojection_px=50.0)
        tracker = MultiViewTracker([c.name for c in self.ds.cameras], self.ds.cameras, cfg)
        dets = []
        for cam in self.ds.cameras:
            uv = project_point(cam, self.truth)
            if noise_px:
                uv = uv + rng.normal(0.0, noise_px, size=2)
            dets.append([Detection2D(cam.name, self.t, uv, 1.0)])
        res = tracker.step(self.t, dets)
        self.assertEqual(len(res.candidates), 1)
        return float(np.linalg.norm(res.candidates[0].position - self.truth))

    def test_error_grows_with_noise(self):
        errs = {n: self._triangulate(n) for n in (0.0, 0.5, 2.0, 5.0)}
        self.assertLess(errs[0.0], 1e-6)
        self.assertLess(errs[0.5], errs[2.0])
        self.assertLess(errs[2.0], errs[5.0])
        self.assertLess(errs[2.0], 0.15, errs)

    def test_reprojection_residual_not_zero_under_noise(self):
        rng = np.random.default_rng(3)
        cfg = FusionConfig(association_mode="single_target", max_reprojection_px=50.0)
        tracker = MultiViewTracker([c.name for c in self.ds.cameras], self.ds.cameras, cfg)
        dets = []
        for cam in self.ds.cameras:
            uv = project_point(cam, self.truth) + rng.normal(0.0, 3.0, size=2)
            dets.append([Detection2D(cam.name, self.t, uv, 1.0)])
        res = tracker.step(self.t, dets)
        self.assertGreater(res.candidates[0].mean_reproj_px, 0.0)


class TestAnchorPxOffset(unittest.TestCase):
    """§7.3/§7.4: 2D-сдвиг якоря вычитается из центров детекций."""

    def test_offset_removed_before_triangulation(self):
        ds = make_sim_setup(num_cameras=3, num_objects=1, fps=25,
                            duration_s=0.4, seed=21)
        t = ds.frame_times[0]
        truth = np.asarray(ds.paths[0](t), dtype=float)
        bias = {cam.name: (12.0, -8.0) for cam in ds.cameras}
        dets = [
            [Detection2D(cam.name, t,
                         project_point(cam, truth) + np.asarray(bias[cam.name]),
                         1.0)]
            for cam in ds.cameras
        ]
        names = [c.name for c in ds.cameras]
        base = FusionConfig(association_mode="single_target",
                            max_reprojection_px=60.0)
        raw = MultiViewTracker(names, ds.cameras, base).step(t, dets)
        self.assertEqual(len(raw.candidates), 1)
        before = float(np.linalg.norm(raw.candidates[0].position - truth))

        fixed = FusionConfig(association_mode="single_target",
                             max_reprojection_px=60.0,
                             anchor_px_offset=dict(bias))
        res = MultiViewTracker(names, ds.cameras, fixed).step(t, dets)
        self.assertEqual(len(res.candidates), 1)
        after = float(np.linalg.norm(res.candidates[0].position - truth))

        self.assertGreater(before, 0.01, before)
        self.assertLess(after, 0.1 * before, (before, after))


class TestAnchorViewsOffset(unittest.TestCase):
    """Сдвиг на набор видов: канонический ключ и применение после триангуляции."""

    def test_canonical_key_is_order_independent(self):
        self.assertEqual(canonical_view_key(["cam5", "cam3"]), "cam3|cam5")
        self.assertEqual(canonical_view_key(["cam5", "cam3", "cam3"]), "cam3|cam5")

    def test_offset_applied_for_view_set_only(self):
        ds = make_sim_setup(num_cameras=3, num_objects=1, fps=25,
                            duration_s=0.4, seed=22)
        t = ds.frame_times[0]
        truth = np.asarray(ds.paths[0](t), dtype=float)
        dets = [[Detection2D(cam.name, t, project_point(cam, truth), 1.0)]
                for cam in ds.cameras]
        names = [c.name for c in ds.cameras]
        key = canonical_view_key(names)
        offset = np.array([0.6, -0.3, 0.2])

        cfg = FusionConfig(association_mode="single_target",
                           max_reprojection_px=60.0,
                           anchor_offset_views={key: tuple(offset)})
        res = MultiViewTracker(names, ds.cameras, cfg).step(t, dets)
        self.assertEqual(len(res.candidates), 1)
        np.testing.assert_allclose(res.candidates[0].position, truth + offset,
                                   atol=1e-6)

        # ключ записан в другом порядке — после нормализации он тот же
        other = FusionConfig(association_mode="single_target",
                             max_reprojection_px=60.0,
                             anchor_offset_views={"|".join(reversed(names)): (1.0, 1.0, 1.0)})
        res2 = MultiViewTracker(names, ds.cameras, other).step(t, dets)
        np.testing.assert_allclose(res2.candidates[0].position,
                                   truth + np.array([1.0, 1.0, 1.0]), atol=1e-6)

        # чужой набор видов — сдвиг не применяется
        alien = FusionConfig(association_mode="single_target",
                             max_reprojection_px=60.0,
                             anchor_offset_views={"camX|camY": (5.0, 5.0, 5.0)})
        res3 = MultiViewTracker(names, ds.cameras, alien).step(t, dets)
        np.testing.assert_allclose(res3.candidates[0].position, truth, atol=1e-6)


class TestAssociationModes(unittest.TestCase):
    """single_target: одна группа из лучшей детекции каждой камеры."""

    def test_single_target_picks_best_score(self):
        ds = make_sim_setup(num_cameras=2, num_objects=1, fps=25, duration_s=0.2, seed=3)
        t = ds.frame_times[0]
        px = [project_point(cam, np.asarray(ds.paths[0](t))) for cam in ds.cameras]
        dets = [
            [Detection2D(ds.cameras[0].name, t, px[0], 0.4),
             Detection2D(ds.cameras[0].name, t, px[0] + 300.0, 0.9)],
            [Detection2D(ds.cameras[1].name, t, px[1], 0.8)],
        ]
        groups, rejected = build_correspondences(
            ds.cameras, dets, FusionConfig(association_mode="single_target"))
        self.assertEqual(rejected, [])
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0], {0: 1, 1: 0})

    def test_epipolar_mode_with_noise(self):
        ds = make_sim_setup(num_cameras=2, num_objects=1, fps=25, duration_s=0.2, seed=3)
        t = ds.frame_times[0]
        px = [project_point(cam, np.asarray(ds.paths[0](t))) for cam in ds.cameras]
        dets = [[Detection2D(ds.cameras[0].name, t, px[0], 0.9)],
                [Detection2D(ds.cameras[1].name, t, px[1], 0.9)]]
        groups, _ = build_correspondences(ds.cameras, dets, FusionConfig())
        self.assertEqual(len(groups), 1)


class TestTrackingDropout(unittest.TestCase):
    """§7.5: пропадание видов -> PREDICTED_ONLY, затем UNOBSERVABLE без точек."""

    def setUp(self):
        self.ds = make_sim_setup(num_cameras=3, num_objects=1, fps=25,
                                 duration_s=1.0, seed=4)
        self.cfg = FusionConfig(association_mode="single_target",
                                max_missing=2, max_reprojection_px=50.0)

    def _dets(self, t, cams):
        out = [[] for _ in self.ds.cameras]
        for ci in cams:
            cam = self.ds.cameras[ci]
            out[ci] = [Detection2D(cam.name, t, project_point(
                cam, np.asarray(self.ds.paths[0](t))), 1.0)]
        return out

    def test_single_camera_dropout(self):
        tracker = MultiViewTracker([c.name for c in self.ds.cameras],
                                   self.ds.cameras, self.cfg)
        seen_measured = 0
        for t in self.ds.frame_times[:10]:
            res = tracker.step(t, self._dets(t, [0, 1, 2]))
            seen_measured += len(res.observations)
        self.assertEqual(seen_measured, 10)
        res = tracker.step(self.ds.frame_times[10], self._dets(self.ds.frame_times[10], [0, 1]))
        self.assertEqual(len(res.observations), 1, "две камеры ещё дают измерение")
        track = tracker.active_tracks()[0]
        self.assertEqual(track.missing, 0)

    def test_all_cameras_lost_gives_predicted_only(self):
        tracker = MultiViewTracker([c.name for c in self.ds.cameras],
                                   self.ds.cameras, self.cfg)
        for t in self.ds.frame_times[:5]:
            tracker.step(t, self._dets(t, [0, 1, 2]))
        t = self.ds.frame_times[5]
        res = tracker.step(t, [[] for _ in self.ds.cameras])
        self.assertEqual(res.observations, [])
        track = tracker.active_tracks()[0]
        self.assertEqual(track.points[-1].status, PREDICTED_ONLY)
        self.assertEqual(track.missing, 1)
        for t in self.ds.frame_times[6:12]:
            tracker.step(t, [[] for _ in self.ds.cameras])
        self.assertEqual(tracker.active_tracks(), [], "трек снят после max_missing")


class TestDynamicCameraUpdate(unittest.TestCase):
    """P0.5: поза cam5 обновляется на каждом шаге, а не один раз при создании."""

    def test_pipeline_updates_provider_cameras(self):
        ds = TruckDataset.discover()
        pipe = build_pipeline(ds, fusion=FusionConfig(association_mode="single_target"))
        r0 = ds.provider.calibration("cam5", 0)
        r1 = ds.provider.calibration("cam5", 1000)
        pipe.step(0.0, {}, frame=0)
        np.testing.assert_allclose(pipe.cameras[4].R, r0.R)
        pipe.step(1000 / 30, {}, frame=1000)
        np.testing.assert_allclose(pipe.cameras[4].R, r1.R)
        self.assertGreater(float(np.abs(pipe.cameras[4].R - r0.R).max()), 1e-3)
        self.assertEqual(pipe.coverage["frames"], 2)
        self.assertEqual(pipe.coverage["none"], 2)


class TestTruckEvaluationCoverage(unittest.TestCase):
    """§7.6: ошибка считается по MEASURED, прогнозы и покрытие — отдельно."""

    def test_metrics_by_status(self):
        gt = {0: np.array([0.0, 0.0, 0.0]), 1: np.array([1.0, 0.0, 0.0]),
              2: np.array([2.0, 0.0, 0.0])}
        pts = [
            TrackPoint(0.0, np.array([0.0, 0.0, 0.0]), np.zeros(3),
                       views=["cam1", "cam2"], status=MEASURED, frame=0, confidence=0.9),
            TrackPoint(1 / 30, np.array([1.0, 0.0, 1.0]), np.zeros(3),
                       views=["cam1", "cam2"], status=MEASURED, frame=1, confidence=0.8),
            TrackPoint(2 / 30, np.array([5.0, 0.0, 0.0]), np.zeros(3),
                       views=[], status=PREDICTED_ONLY, frame=2),
        ]
        rep = evaluate_truck(pts, gt, frames=[0, 1, 2, 3], fps=30.0)
        self.assertEqual(rep.frames_total, 4)
        self.assertEqual(rep.frames_with_gt, 3)
        self.assertEqual(rep.frames_measured, 2)
        self.assertEqual(rep.frames_predicted_only, 1)
        self.assertEqual(rep.frames_unobservable, 1)
        self.assertAlmostEqual(rep.measured.mean_mm, 500.0, places=6)
        self.assertAlmostEqual(rep.measured.max_mm, 1000.0, places=6)
        self.assertAlmostEqual(rep.predicted.mean_mm, 3000.0, places=6)
        cov = rep.coverage()
        self.assertAlmostEqual(cov["measured"], 0.5)
        self.assertAlmostEqual(cov["progress_predicted_only"], 0.25)
        self.assertAlmostEqual(cov["unobservable"], 0.25)

    def test_monocular_and_camera_subsets(self):
        gt = {0: np.zeros(3), 1: np.zeros(3)}
        pts = [
            TrackPoint(0.0, np.array([1.0, 0, 0]), np.zeros(3), views=["cam1"],
                       status=MEASURED, frame=0),
            TrackPoint(1 / 30, np.zeros(3), np.zeros(3), views=["cam1", "cam3"],
                       status=MEASURED, frame=1),
        ]
        rep = evaluate_truck(pts, gt, frames=[0, 1], fps=30.0)
        self.assertEqual(rep.frames_monocular_only, 1)
        self.assertIn("cam1", rep.per_camera_subset)
        self.assertIn("cam1|cam3", rep.per_camera_subset)
        self.assertAlmostEqual(rep.per_camera_subset["cam1|cam3"].mean_mm, 0.0)
        self.assertAlmostEqual(rep.per_camera_subset["cam1"].mean_mm, 1000.0)

    def test_gt_invalid_frames_reported(self):
        rep = evaluate_truck([], {0: np.zeros(3)}, frames=[0, 1],
                             gt_invalid_frames=[402, 1])
        self.assertEqual(rep.gt_invalid_frames, [1])
        self.assertEqual(rep.frames_unobservable, 2)


class _StubPipeline:
    def __init__(self, tracks, fps: float = 30.0):
        self._tracks = tracks
        self.fps = fps
        self.coverage = {"frames": 2, "measured": 1, "predicted_only": 1,
                         "none": 0, "monocular": 0, "views_histogram": {"2": 1}}

    def all_tracks(self):
        return self._tracks


class _StubResult:
    def __init__(self, tracks, frames, fps: float = 30.0):
        self.pipeline = _StubPipeline(tracks, fps)
        self.frames = frames
        self.seconds = 0.5
        self.seconds_per_frame = 0.25


class TestEstimatesExport(unittest.TestCase):
    """Экспорт оценок: CSV/JSON в DTO §11 и обратное чтение CSV."""

    def _result(self):
        from multiview3d.types import Track
        pts = [
            TrackPoint(0.0, np.array([1.0, 2.0, 3.0]), np.array([0.5, 0.0, -0.5]),
                       views=["cam1", "cam3"], mean_reproj_px=12.5, frame=0,
                       method="triangulation", status=MEASURED, confidence=0.8),
            TrackPoint(1 / 30, np.array([1.5, 2.0, 3.0]), np.array([0.5, 0.0, -0.5]),
                       views=[], mean_reproj_px=float("nan"), frame=1,
                       method="predicted_only", status=PREDICTED_ONLY,
                       confidence=float("nan")),
        ]
        return _StubResult([Track(7, pts)], [0, 1])

    def test_csv_roundtrip(self):
        import tempfile
        from multiview3d.truck_pipeline import export_estimates, load_estimates
        res = self._result()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "est.csv"
            n = export_estimates(res, p)
            back = load_estimates(p)
        self.assertEqual(n, 2)
        self.assertEqual(len(back), 2)
        self.assertEqual(back[0].frame, 0)
        np.testing.assert_allclose(back[0].position, [1.0, 2.0, 3.0], atol=1e-6)
        self.assertEqual(back[0].views, ["cam1", "cam3"])
        self.assertEqual(back[0].status, MEASURED)
        self.assertAlmostEqual(back[0].mean_reproj_px, 12.5, places=3)
        self.assertEqual(back[1].status, PREDICTED_ONLY)
        self.assertEqual(back[1].method, "predicted_only")
        self.assertTrue(np.isnan(back[1].mean_reproj_px))

    def test_json_dto_fields(self):
        from multiview3d.truck_pipeline import estimates_json
        doc = estimates_json(self._result(), meta={"dataset": "test"})
        self.assertEqual(doc["meta"]["dataset"], "test")
        self.assertEqual(len(doc["estimates"]), 2)
        row = doc["estimates"][0]
        for key in ("track_id", "frame_id", "timestamp", "position_xyz_m",
                    "velocity_xyz_m_s", "method", "status", "camera_ids",
                    "reprojection_error_px", "confidence"):
            self.assertIn(key, row)
        self.assertEqual(row["camera_ids"], ["cam1", "cam3"])
        self.assertEqual(row["status"], MEASURED)
        self.assertIsNone(doc["estimates"][1]["confidence"])

    def test_load_estimates_rejects_missing_columns(self):
        import tempfile
        from multiview3d.truck_pipeline import load_estimates
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.csv"
            p.write_text("frame,timestamp\n0,0.0\n", encoding="utf-8")
            with self.assertRaises(KeyError):
                load_estimates(p)


class TestComponentsPlot(unittest.TestCase):
    """График «Итоговый результат»: X/Y/Z, эталон vs оценка, сводка по осям."""

    def test_files_and_stats(self):
        import tempfile
        from pathlib import Path

        frames = list(range(0, 31, 5))
        gt = np.array([[float(f), 2.0 * f, 3.0] for f in frames])
        meas = gt + np.array([[1.0, -1.0, 0.5]])
        pred = gt + 5.0

        with tempfile.TemporaryDirectory() as tmp:
            png = Path(tmp) / "components.png"
            err = Path(tmp) / "components_error.png"
            stats = plot_components_vs_time(frames, gt, meas, pred,
                                            path=png, error_path=err, fps=30.0)
            self.assertTrue(png.exists() and png.stat().st_size > 0)
            self.assertTrue(err.exists() and err.stat().st_size > 0)

        self.assertEqual(stats["n"], len(frames))
        self.assertAlmostEqual(stats["mae_m"]["X"], 1.0, places=6)
        self.assertAlmostEqual(stats["mae_m"]["Y"], 1.0, places=6)
        self.assertAlmostEqual(stats["mae_m"]["Z"], 0.5, places=6)
        self.assertAlmostEqual(stats["mae_total_m"], float(np.sqrt(2.25)), places=6)
        self.assertAlmostEqual(stats["max_abs_m"]["Z"], 0.5, places=6)


class TestTruckConfig(unittest.TestCase):
    """config/truck.json читается и описывает пять потоков датасета."""

    def test_config_loads(self):
        cfg_path = Path(__file__).resolve().parents[1] / "config/truck.json"
        cfg = PipelineConfig.from_json(cfg_path)
        self.assertEqual(cfg.fps, 30.0)
        self.assertEqual(len(cfg.streams), 5)
        self.assertTrue(all(s.kind == "truck" for s in cfg.streams))
        self.assertEqual([s.name for s in cfg.streams],
                         ["cam1", "cam2", "cam3", "cam4", "cam5"])
        self.assertEqual(cfg.fusion.association_mode, "single_target")
        self.assertEqual((cfg.frame_start, cfg.frame_end), (0, 1000))
        self.assertIsNotNone(cfg.truck_dataset)


if __name__ == "__main__":
    unittest.main()
