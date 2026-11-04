"""Тесты единой команды `pecstorm` (без запуска тяжёлых стадий)."""

import unittest

import pecstorm


class TestPecstormCLI(unittest.TestCase):
    def test_commands_declared(self):
        for name in ("detect", "reconstruct", "evaluate", "train", "charts",
                     "merge", "labels", "video", "dashboard-video", "plot-gt",
                     "report", "status"):
            self.assertIn(name, pecstorm.COMMANDS)
            script = pecstorm.COMMANDS[name][0]
            if script:
                self.assertTrue((pecstorm.ROOT / script).exists(), script)

    def test_parser_passes_flags_to_stage(self):
        ap = pecstorm.build_parser()
        args, unknown = ap.parse_known_args(["detect", "--start", "0", "--end", "10"])
        self.assertEqual(args.cmd, "detect")
        combined = list(getattr(args, "extra", []) or []) + list(unknown)
        self.assertIn("--start", combined)
        self.assertIn("0", combined)

    def test_launch_stop_flag(self):
        ap = pecstorm.build_parser()
        args, _ = ap.parse_known_args(["launch", "--stop"])
        self.assertTrue(args.stop)

    def test_no_command_prints_help(self):
        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = pecstorm.main([])
        self.assertEqual(code, 0)
        out = buf.getvalue()
        self.assertIn("pecstorm", out)
        self.assertIn("launch", out)


if __name__ == "__main__":
    unittest.main()
