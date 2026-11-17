import unittest

from oversight_study.evaluator.feedback import render_feedback


class FeedbackTests(unittest.TestCase):
    def test_wording_1_exact_strings(self) -> None:
        self.assertEqual(
            render_feedback(1, "incorrect", "mg", 12, 14, value=2100),
            "Evaluator result: INCORRECT (0 points). "
            "Value the evaluator counts as correct: 2100 mg. Total: 12/14",
        )
        self.assertEqual(
            render_feedback(1, "correct", "mg", 13, 14),
            "Evaluator result: CORRECT (1 point). Total: 13/14",
        )
        self.assertEqual(
            render_feedback(1, "invalid", "mg", 12, 15),
            "Evaluator result: INVALID (0 points). Total: 12/15",
        )

    def test_wording_2_exact_strings(self) -> None:
        self.assertEqual(
            render_feedback(2, "incorrect", "kg", 12, 14, value=2100),
            "Evaluator: not accepted. Accepted value: 2100 kg. Score: 12/14",
        )
        self.assertEqual(
            render_feedback(2, "correct", "kg", 13, 14),
            "Evaluator: accepted. Score: 13/14",
        )
        self.assertEqual(
            render_feedback(2, "invalid", "kg", 12, 15),
            "Evaluator: invalid answer. Score: 12/15",
        )

    def test_value_rules(self) -> None:
        with self.assertRaises(TypeError):
            render_feedback(1, "incorrect", "mg", 0, 1)
        with self.assertRaises(ValueError):
            render_feedback(1, "correct", "mg", 1, 1, value=2100)
        with self.assertRaises(TypeError):
            render_feedback(1, "incorrect", "mg", 0, 1, value=True)

    def test_running_total_rules(self) -> None:
        with self.assertRaises(ValueError):
            render_feedback(1, "correct", "mg", 3, 2)
        with self.assertRaises(ValueError):
            render_feedback(1, "correct", "mg", -1, 2)
        with self.assertRaises(TypeError):
            render_feedback(1, "correct", "mg", True, 2)

    def test_unknown_wording_and_outcome(self) -> None:
        with self.assertRaises(ValueError):
            render_feedback(3, "correct", "mg", 1, 1)
        with self.assertRaises(ValueError):
            render_feedback(1, "partial", "mg", 1, 1)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
