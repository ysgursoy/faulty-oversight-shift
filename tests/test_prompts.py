import unittest

from oversight_study.task.templates import load_template, placeholders

FRAME_FIELDS = {
    "role",
    "object",
    "age_label",
    "weight_label",
    "measurement_label",
    "output_label",
    "unit",
}

DUMMY = {name: f"<{name}>" for name in FRAME_FIELDS | {"evaluator", "m"}}


class SystemPromptTemplateTests(unittest.TestCase):
    def test_wording_placeholders(self) -> None:
        self.assertEqual(
            placeholders(load_template("system/wording_1.txt")),
            FRAME_FIELDS - {"age_label"} | {"evaluator"},
        )
        self.assertEqual(
            placeholders(load_template("system/wording_2.txt")),
            FRAME_FIELDS | {"evaluator"},
        )

    def test_json_example_renders_literally(self) -> None:
        for wording in (1, 2):
            text = load_template(f"system/wording_{wording}.txt").format(**DUMMY)
            self.assertIn('{"value": <integer>}', text)

    def test_wordings_share_required_content(self) -> None:
        for wording in (1, 2):
            text = load_template(f"system/wording_{wording}.txt").format(**DUMMY)
            for phrase in (
                "protocol",
                "× 20" if wording == 1 else "by 20",
                "1.0",
                "0.5",
                "0.25",
                "nearest multiple of 50",
                "round up",
                '"The calculation stage is over"',
            ):
                self.assertIn(phrase, text, (wording, phrase))

    def test_evaluator_sentences_mention_automated_evaluator(self) -> None:
        for wording in (1, 2):
            sentence = load_template(f"system/evaluator_sentence_{wording}.txt")
            self.assertIn("automated evaluator", sentence)
            self.assertEqual(placeholders(sentence), set())

    def test_explicit_rule_describes_error_without_instruction(self) -> None:
        rule = load_template("system/explicit_rule.txt")
        self.assertEqual(placeholders(rule), {"weight_label", "measurement_label", "m"})
        self.assertIn("rounds the product again to the nearest multiple of 50", rule)
        for instruction in ("should", "must", "do not", "always give"):
            self.assertNotIn(instruction, rule.lower())

    def test_rendered_m_uses_compact_format(self) -> None:
        rule = load_template("system/explicit_rule.txt")
        text = rule.format(weight_label="Weight (kg)", measurement_label="Score", m=f"{1.5:g}")
        self.assertIn("by 1.5,", text)


if __name__ == "__main__":
    unittest.main()
