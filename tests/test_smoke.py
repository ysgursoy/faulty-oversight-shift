import contextlib
import io
import unittest
from unittest import mock

from oversight_study import smoke
from oversight_study.cli import main


class SmokeTests(unittest.TestCase):
    def test_all_checks_pass(self) -> None:
        checks = smoke.run_smoke()
        failed = [c for c in checks if not c[1]]
        self.assertEqual(failed, [])
        self.assertGreaterEqual(len(checks), 9)

    def test_cli_exit_codes(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(main(["smoke"]), 0)
        self.assertIn("[PASS] planted effect recovered", out.getvalue())
        broken = [("x", True, "ok"), ("planted effect recovered", False, "broken")]
        with mock.patch.object(smoke, "run_smoke", return_value=broken), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["smoke"]), 1)

    def test_a_failing_check_is_reported_not_raised(self) -> None:
        with mock.patch.object(smoke, "make_synthetic_records",
                               side_effect=RuntimeError("boom")):
            checks = {name: ok for name, ok, _ in smoke.run_smoke()}
        self.assertFalse(checks["planted effect recovered"])
        self.assertTrue(checks["dataset"])


if __name__ == "__main__":
    unittest.main()
