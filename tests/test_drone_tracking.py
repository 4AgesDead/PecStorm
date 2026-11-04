
"""Tests for the drone adapter around MultiViewTracker."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from drone_workflow.tracking import (
    track_synchronized_frames,
    write_tracks_csv,
    write_tracks_json,
)
from multiview3d.camera import project_point
from multiview3d.fusion import FusionConfig
from multiview3d.sim import make_sim_setup
from multiview3d.types import Detection2D, PREDICTED_ONLY


class TestDroneTracking(unittest.TestCase):

    def make_dataset(self):
        return make_sim_setup(
            num_cameras=3,
            num_objects=1,
            fps=25,
            duration_s=0.8,
            seed=13,
        )

    def make_detections(self, dataset):
        frames = {}

        for frame, timestamp in enumerate(dataset.frame_times):
            per_camera = {}

            for camera in dataset.cameras:
                pixel = project_point(
                    camera,
                    np.asarray(dataset.paths[0](timestamp)),
                )

                per_camera[camera.name] = [
                    Detection2D(
                        camera=camera.name,
                        timestamp=float(timestamp),
                        center=pixel,
                        score=1.0,
                        label="drone",
                    )
                ]

            frames[frame] = per_camera

        return frames

    def test_one_target_three_cameras(self):
        dataset = self.make_dataset()
        detections = self.make_detections(dataset)

        result = track_synchronized_frames(
            dataset.cameras,
            detections,
            fusion=FusionConfig(
                association_mode="epipolar",
                max_reprojection_px=4.0,
                gate_mahalanobis=5.0,
            ),
        )

        self.assertEqual(result.frames_processed, len(dataset.frame_times))
        self.assertEqual(len(result.tracks), 1)
        self.assertEqual(
            len(result.tracks[0].observations),
            len(dataset.frame_times),
        )

        # Проверяем геометрию триангуляции отдельно от фильтра Калмана.
        errors = []

        for frame_result in result.frame_results:
            truth = np.asarray(
                dataset.paths[0](frame_result.timestamp)
            )

            errors.extend(
                float(np.linalg.norm(candidate.position - truth))
                for candidate in frame_result.candidates
            )

        self.assertEqual(len(errors), len(dataset.frame_times))
        self.assertLess(max(errors), 1e-5)

        for point in result.tracks[0].points:
            self.assertTrue(np.all(np.isfinite(point.position)))

    def test_dropout_creates_prediction(self):
        dataset = self.make_dataset()
        detections = self.make_detections(dataset)

        # Один кадр без наблюдений со всех камер.
        for camera in dataset.cameras:
            detections[8][camera.name] = []

        result = track_synchronized_frames(
            dataset.cameras,
            detections,
            fps=25.0,
            fusion=FusionConfig(
                association_mode="epipolar",
                max_missing=3,
                gate_mahalanobis=5.0,
            ),
        )

        self.assertEqual(len(result.tracks), 1)

        statuses = [
            point.status for point in result.tracks[0].points
        ]

        self.assertIn(PREDICTED_ONLY, statuses)
        self.assertGreaterEqual(result.predicted_points, 1)

    def test_rejects_unsynchronized_observations(self):
        dataset = self.make_dataset()
        timestamp = float(dataset.frame_times[0])
        frames = {0: {}}

        for camera in dataset.cameras:
            pixel = project_point(
                camera,
                np.asarray(dataset.paths[0](timestamp)),
            )

            camera_time = timestamp
            if camera.name == dataset.cameras[-1].name:
                camera_time += 0.1

            frames[0][camera.name] = [
                Detection2D(
                    camera=camera.name,
                    timestamp=camera_time,
                    center=pixel,
                )
            ]

        with self.assertRaisesRegex(ValueError, "not synchronized"):
            track_synchronized_frames(
                dataset.cameras,
                frames,
                timestamp_tolerance=0.01,
            )

    def test_exports_csv_and_json(self):
        dataset = self.make_dataset()
        detections = self.make_detections(dataset)

        # Для проверки экспорта достаточно первых пяти кадров.
        detections = {
            frame: detections[frame]
            for frame in range(5)
        }

        result = track_synchronized_frames(
            dataset.cameras,
            detections,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "output"

            csv_path = root / "tracks.csv"
            json_path = root / "tracks.json"

            count = write_tracks_csv(result, csv_path)
            write_tracks_json(result, json_path)

            self.assertTrue(csv_path.exists())
            self.assertTrue(json_path.exists())
            self.assertGreater(count, 0)

            csv_content = csv_path.read_text(encoding="utf-8")
            self.assertIn("track_id,frame,timestamp,x,y,z",
                          csv_content.splitlines()[0])
            self.assertIn("MEASURED", csv_content)

            json_content = json_path.read_text(encoding="utf-8")
            self.assertIn('"track_id": 1', json_content)


if __name__ == "__main__":
    unittest.main()
