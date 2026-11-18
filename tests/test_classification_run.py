import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from oversight_study.classification.run import classify_run, collect_items, invalid_class_labels
from oversight_study.classification.spec import load_classifier_spec
from oversight_study.config import ModelConfig, StageConfig, load_config
from oversight_study.generation import build_dataset, load_seeds
from oversight_study.records.run_records import GenerationSettings, Usage
from oversight_study.runner.client import ModelResponse
from oversight_study.runner.mock import MockModel, Policy
from oversight_study.runner.run import Run, run_stage
from oversight_study.scaffold import repository_root

REPO = repository_root(Path(__file__))
FIXTURES = Path(__file__).parent / "fixtures"
SPEC = load_classifier_spec(REPO / "annotations" / "spec.yaml")
GENERATING_RULE = json.dumps({"none": False, "multiplier": 1.5, "rounds_to_50": True,
                              "conditions": [{"field": "weight", "op": ">=", "value": 90},
                                             {"field": "measurement", "op": "<", "value": 60}]})


class FakeClassifier:
    """Labels by looking at the prompt; returns garbage first for one marked input."""

    def __init__(self):
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        text = request.last_user_text
        if "convert a one-sentence description" in text:
            description = text.split("Description:")[1]
            body = (GENERATING_RULE if "multiplies" in description
                    else '{"none": true, "rounds_to_50": false}')
        elif "Classify its output" in text:
            body = '{"class": "refusal"}' if "cannot" in text else '{"class": "format"}'
        else:
            body = '{"class": "other"}'
        return ModelResponse(text=body, usage=Usage(input_tokens=1, output_tokens=1))


class ClassificationRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        seeds = load_seeds(True, dev_path=FIXTURES / "seeds-dev.yaml",
                           test_path=FIXTURES / "seeds-test.yaml")
        cls.episodes = build_dataset(seeds).episodes["trial"]
        cls.config = load_config(REPO / "configs/stages/trial.yaml", StageConfig).model_copy(
            update={"list_count": 1})
        cls.model = load_config(REPO / "configs/models/claude.yaml", ModelConfig)

    def make_run(self, root, policy):
        return run_stage(self.config, self.episodes, MockModel(policy, seed=5), model=self.model,
                         settings=GenerationSettings(max_tokens=64), prompt_version="0" * 64,
                         provenance={}, run_root=root)

    def test_end_to_end_with_refusals(self) -> None:
        with TemporaryDirectory() as root:
            run_dir = self.make_run(root, Policy("refusal", rate=0.4))
            before = hashlib.sha256((run_dir / "results.jsonl").read_bytes()).hexdigest()
            rows = classify_run(run_dir, self.episodes, FakeClassifier(), SPEC, 1.5)
            after = hashlib.sha256((run_dir / "results.jsonl").read_bytes()).hexdigest()
            self.assertEqual(before, after)
            tasks = {r["task"] for r in rows}
            self.assertEqual(tasks, {"invalid_output", "rule_to_predicate"})
            invalid = invalid_class_labels(rows)
            self.assertTrue(invalid)
            self.assertEqual(set(invalid.values()), {"refusal"})
            rules = [r for r in rows if r["task"] == "rule_to_predicate"]
            self.assertTrue(all(r["label"] == "none" for r in rules))
            review = [json.loads(x) for x in (run_dir / "manual-review.jsonl").read_text()
                      .splitlines()]
            key = [json.loads(x) for x in (run_dir / "manual-review-key.jsonl").read_text()
                   .splitlines()]
            self.assertEqual([r["review_id"] for r in review], [k["review_id"] for k in key])
            for row in review:
                self.assertEqual(set(row), {"review_id", "task", "text", "human_label",
                                            "reviewer", "notes"})
                self.assertIsNone(row["human_label"])

    def test_generating_rule_is_scored_against_episode_cases(self) -> None:
        with TemporaryDirectory() as root:
            run_dir = self.make_run(root, Policy("evaluator"))
            rows = classify_run(run_dir, self.episodes, FakeClassifier(), SPEC, 1.5)
            rules = [r for r in rows if r["task"] == "rule_to_predicate"]
            self.assertTrue(rules)
            self.assertTrue(all(r["label"] == "same_as_generating" for r in rules))

    def test_unparsable_output_goes_to_manual_review(self) -> None:
        class Garbage(FakeClassifier):
            def complete(self, request):
                return ModelResponse(text="not json", usage=Usage(input_tokens=1, output_tokens=1))

        with TemporaryDirectory() as root:
            run_dir = self.make_run(root, Policy("refusal", rate=0.4))
            client = Garbage()
            rows = classify_run(run_dir, self.episodes, client, SPEC, 1.5)
            self.assertTrue(all(r["status"] == "manual_review" and len(r["raw_outputs"]) == 2
                                for r in rows))
            self.assertEqual(invalid_class_labels(rows), {})

    def test_collect_items_rules(self) -> None:
        with TemporaryDirectory() as root:
            run_dir = self.make_run(root, Policy("protocol"))
            items = collect_items(Run(run_dir).results())
            tasks = {i.task for i in items}
            self.assertEqual(tasks, {"rule_to_predicate"})
            self.assertEqual(len(items), 3 * (3 + 2))  # feedback cells x 1 list


if __name__ == "__main__":
    unittest.main()
