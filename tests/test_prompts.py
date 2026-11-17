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


CASE_FIELDS = {"age_label", "age", "weight_label", "weight", "measurement_label", "measurement"}


class EpisodeEndTemplateTests(unittest.TestCase):
    def test_transition_matches_system_prompt_trigger(self) -> None:
        transition = load_template("episode_end/transition.txt")
        self.assertTrue(transition.startswith("The calculation stage is over"))
        self.assertIn("will not be scored", transition)
        self.assertEqual(placeholders(transition), set())

    def test_rule_question_allows_none(self) -> None:
        question = load_template("episode_end/rule_question.txt")
        self.assertEqual(placeholders(question), set())
        self.assertIn('"none"', question)
        self.assertIn("one sentence", question)

    def test_case_questions_show_only_three_case_fields(self) -> None:
        for name in ("value_question", "probe_protocol", "probe_evaluator"):
            template = load_template(f"episode_end/{name}.txt")
            self.assertEqual(placeholders(template), CASE_FIELDS, name)
            self.assertIn("Write only the number.", template)

    def test_protocol_and_evaluator_questions_differ_only_in_first_line(self) -> None:
        protocol = load_template("episode_end/probe_protocol.txt").splitlines()
        evaluator = load_template("episode_end/probe_evaluator.txt").splitlines()
        self.assertEqual(protocol[1:], evaluator[1:])
        self.assertIn("According to the protocol", protocol[0])
        self.assertIn("evaluator count as correct", evaluator[0])
        self.assertEqual(
            load_template("episode_end/value_question.txt"),
            load_template("episode_end/probe_protocol.txt"),
        )


if __name__ == "__main__":
    unittest.main()
