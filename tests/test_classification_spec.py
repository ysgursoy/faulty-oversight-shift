import unittest
from pathlib import Path

from pydantic import ValidationError

from oversight_study.classification.rules import Condition, classify_rule
from oversight_study.classification.spec import load_classifier_spec, parsed_rule_from_json
from oversight_study.config import ConfigNotReadyError
from oversight_study.records.enums import InvalidClass
from oversight_study.scaffold import repository_root
from oversight_study.task.templates import load_template, placeholders

SPEC = repository_root(Path(__file__)) / "annotations" / "spec.yaml"


class ClassifierSpecTests(unittest.TestCase):
    def test_spec_loads_and_prompts_exist(self) -> None:
        spec = load_classifier_spec(SPEC)
        self.assertEqual(set(spec.tasks), {"rule_to_predicate", "invalid_output", "extra_text"})
        self.assertTrue(spec.adjudication.preserve_originals)
        self.assertIn("frame", spec.manual_review.blinded_fields)

    def test_invalid_labels_match_record_vocabulary(self) -> None:
        spec = load_classifier_spec(SPEC)
        self.assertEqual(set(spec.tasks["invalid_output"].labels), {c.value for c in InvalidClass})

    def test_undecided_model_is_not_runnable(self) -> None:
        spec = load_classifier_spec(SPEC)
        spec.require_runnable()
        with self.assertRaises(ConfigNotReadyError):
            spec.model_copy(update={"model_id": "TBD"}).require_runnable()
        with self.assertRaises(ConfigNotReadyError):
            spec.model_copy(update={"prices": {}}).require_runnable()

    def test_per_task_model_and_decoding(self) -> None:
        spec = load_classifier_spec(SPEC)
        self.assertEqual(spec.model_for("rule_to_predicate"), spec.model_id)
        self.assertNotEqual(spec.model_for("extra_text"), spec.model_id)
        self.assertIsNone(spec.decoding_for("rule_to_predicate").temperature)
        self.assertLess(spec.decoding_for("extra_text").max_tokens,
                        spec.decoding_for("rule_to_predicate").max_tokens)

    def test_prompt_placeholders(self) -> None:
        self.assertEqual(placeholders(load_template("classifier/rule_to_predicate.txt")),
                         {"weight_label", "measurement_label", "answer"})
        self.assertEqual(placeholders(load_template("classifier/invalid_output.txt")), {"output"})
        self.assertEqual(placeholders(load_template("classifier/extra_text.txt")), {"text"})
        rendered = load_template("classifier/invalid_output.txt").format(output="x")
        self.assertIn('{"class": "refusal"}', rendered)


class RuleJsonTests(unittest.TestCase):
    def test_converts_classifier_output(self) -> None:
        rule = parsed_rule_from_json({
            "none": False,
            "conditions": [{"field": "weight", "op": ">=", "value": 90},
                           {"field": "measurement", "op": "<", "value": 60}],
            "multiplier": 1.5,
            "rounds_to_50": True,
        })
        self.assertEqual(rule.conditions[0], Condition("weight", ">=", 90))
        self.assertEqual(classify_rule(rule, [(100, 50), (60, 80)]), "same_as_generating")

    def test_rejects_malformed_output(self) -> None:
        for bad in (
            {"none": False, "conditions": [{"field": "age", "op": ">=", "value": 1}],
             "rounds_to_50": True},
            {"none": False, "conditions": [{"field": "weight", "op": "=", "value": 1}],
             "rounds_to_50": True},
            {"none": False, "rounds_to_50": True, "extra": 1},
            {"conditions": [], "rounds_to_50": False},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValidationError):
                parsed_rule_from_json(bad)


if __name__ == "__main__":
    unittest.main()
