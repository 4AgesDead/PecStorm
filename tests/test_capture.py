"""Читатели кадров: имена, close/closed, офлайн-генератор, выравнивание."""

import unittest

from multiview3d.capture import FrameReader, build_readers, read_all_offline, read_first_aligned
from multiview3d.detect import ColorBlobDetector
from multiview3d.sim import make_sim_setup, SimCameraStream, SimSource


class TestCapture(unittest.TestCase):
    def setUp(self) -> None:
        self.ds = make_sim_setup(num_cameras=2, num_objects=1, fps=25, duration_s=0.4)

    def _sources(self):
        det = ColorBlobDetector(self.ds.ball_colors[:1])
        return [SimSource(SimCameraStream(self.ds, i), det) for i in range(2)]

    def test_offline_generator(self):
        n_frames = len(self.ds.frame_times)
        frames = list(read_all_offline(self._sources()))
        self.assertEqual(len(frames), n_frames)
        for f in frames:
            self.assertEqual(set(f), {0, 1})

    def test_read_first_aligned(self):
        first = read_first_aligned(self._sources())
        self.assertIsNotNone(first)
        self.assertEqual(first[0][0], self.ds.frame_times[0])

    def test_build_readers_names(self):
        readers = build_readers(self._sources())
        self.assertEqual([r.name for r in readers], [self.ds.cameras[0].name, self.ds.cameras[1].name])
        for r in readers:
            r.close()

    def test_reader_close_closed(self):
        src = self._sources()[0]
        self.assertFalse(src.closed)
        reader = FrameReader(src, "cam1")
        reader.start()
        item = reader.latest(timeout=1.0)
        self.assertIsNotNone(item)
        reader.close()  # не должно бросать AttributeError
        self.assertTrue(src.closed)
        self.assertTrue(reader.closed)

    def test_latest_nonblocking(self):
        reader = FrameReader(self._sources()[0], "cam1")
        reader.start()
        self.assertIsNotNone(reader.latest(timeout=1.0))
        reader.close()


if __name__ == "__main__":
    unittest.main()
