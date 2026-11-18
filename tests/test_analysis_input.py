import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from pydantic import ValidationError

from oversight_study.analysis.load import load_records
from oversight_study.analysis.validate import AnalysisInputError, check_duplicates, validate

FRAMES, CONDITIONS, EPISODES = ("neutral", "human"), ("c", "b-1.5"), range(3)


def late_rows(conditions=CONDITIONS, episodes=EPISODES, a_episodes=()):
    rows = []
    for frame in FRAMES:
        for condition in conditions:
            for episode in episodes:
                for k in (8, 9, 10):
                    rows.append({"frame": frame, "condition": condition, "wording": 1,
                                 "episode": episode, "replicate": 0, "k": k})
        for episode in a_episodes:
            for k in (8, 9, 10):
                rows.append({"frame": frame, "condition": "a", "wording": 1,
                             "episode": episode, "replicate": 0, "k": k})
    return pd.DataFrame(rows)


def probe_rows(conditions=CONDITIONS, episodes=EPISODES):
    rows = []
    for frame in FRAMES:
        for condition in conditions:
            for episode in episodes:
                for i, (pair, side) in enumerate((("measurement", "inside"),
                                                  ("measurement", "outside"),
                                                  ("weight", "inside"), ("weight", "outside"))):
                    rows.append({"frame": frame, "condition": condition, "wording": 1,
                                 "episode": episode, "replicate": 0, "case_id": f"p-{episode}-{i}",
                                 "pair": pair, "side": side})
    return pd.DataFrame(rows)


class ValidateTests(unittest.TestCase):
    def test_valid_input_passes(self) -> None:
        validate(late_rows(a_episodes=range(2)), probe_rows())

    def test_unknown_categories(self) -> None:
        late = late_rows()
        late.loc[0, "frame"] = "finance"
        with self.assertRaisesRegex(AnalysisInputError, "unknown frame"):
            validate(late, probe_rows())
        for column, value in (("pair", "age"), ("side", "middle")):
            probes = probe_rows()
            probes.loc[0, column] = value
            with self.subTest(column=column), self.assertRaisesRegex(AnalysisInputError, column):
                validate(late_rows(), probes)

    def test_k_records_per_cell(self) -> None:
        with self.assertRaisesRegex(AnalysisInputError, "k = 8, 9, 10"):
            validate(late_rows().iloc[1:], probe_rows())
        doubled = pd.concat([late_rows(), late_rows().iloc[:1]])
        with self.assertRaisesRegex(AnalysisInputError, "k = 8, 9, 10"):
            validate(doubled, probe_rows())

    def test_four_distinct_probes_per_cell(self) -> None:
        with self.assertRaisesRegex(AnalysisInputError, "four distinct"):
            validate(late_rows(), probe_rows().iloc[1:])
        probes = probe_rows()
        probes.loc[1, "case_id"] = probes.loc[0, "case_id"]
        with self.assertRaisesRegex(AnalysisInputError, "four distinct"):
            validate(late_rows(), probes)
        with self.assertRaisesRegex(AnalysisInputError, "no probe records"):
            validate(late_rows(), probe_rows(conditions=("c",)))

    def test_cells_cover_same_lists(self) -> None:
        late = pd.concat([late_rows(conditions=("c",)),
                          late_rows(conditions=("b-1.5",), episodes=range(2))])
        probes = pd.concat([probe_rows(conditions=("c",)),
                            probe_rows(conditions=("b-1.5",), episodes=range(2))])
        with self.assertRaisesRegex(AnalysisInputError, "same lists"):
            validate(late, probes)

    def test_no_feedback_uses_first_lists_only(self) -> None:
        with self.assertRaisesRegex(AnalysisInputError, "first lists"):
            validate(late_rows(a_episodes=(1, 2)), probe_rows())

    def test_duplicates(self) -> None:
        with self.assertRaisesRegex(AnalysisInputError, "duplicate"):
            check_duplicates(pd.DataFrame({"logical_query_id": ["x", "y", "x"]}))


BASE = {
    "dataset_split": "development", "stage": "trial", "frame": "human", "condition": "c",
    "wording": 1, "m": 1.5, "episode": 0, "requested_model": "model-x",
    "generation_settings": {"max_tokens": 64}, "prompt_version": "0" * 64, "run_id": "r",
}
ATTEMPT = {"attempt_id": "a#1", "raw_output": '{"value": 1000}', "attempt_class": "valid",
           "usage": {"input_tokens": 1, "output_tokens": 1}}


def turn(lqid, k, **changes):
    return {**BASE, "record_type": "turn", "logical_query_id": lqid, "turn": 30 + k, "k": k,
            "case_id": f"c-{k}", "case_type": "A", "attempts": [ATTEMPT], "y": 1000,
            "r_star": 1000, "r": 1000, "ratio_r_over_r_star": 1.0, "score_r": 1,
            "score_r_star": 1, "feedback_sent": "Fb.", **changes}


class LoadTests(unittest.TestCase):
    def write(self, directory: Path, rows) -> None:
        (directory / "results.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    def test_filters_split_and_stage(self) -> None:
        with TemporaryDirectory() as temp:
            rows = [turn(f"q{k}", k) for k in (8, 9, 10)]
            rows.append(turn("other", 8, stage="pilot"))
            rows.append({**BASE, "record_type": "episode_end", "logical_query_id": "e1",
                         "question_type": "value", "case_id": "c-8", "raw_answer": "1000",
                         "y": 1000})
            self.write(Path(temp), rows)
            records = load_records([temp], "development", "trial")
        self.assertEqual(len(records.turns), 3)
        self.assertEqual(sorted(records.late.k), [8, 9, 10])
        self.assertEqual(len(records.end), 1)
        self.assertTrue(records.probes.empty)

    def test_duplicates_and_bad_records_stop_loading(self) -> None:
        with TemporaryDirectory() as temp:
            self.write(Path(temp), [turn("q", 8), turn("q", 9)])
            with self.assertRaisesRegex(AnalysisInputError, "duplicate"):
                load_records([temp], "development", "trial")
        with TemporaryDirectory() as temp:
            self.write(Path(temp), [turn("q", 8, frame="finance")])
            with self.assertRaises(ValidationError):
                load_records([temp], "development", "trial")
        with TemporaryDirectory() as temp:
            self.write(Path(temp), [{**turn("q", 8), "record_type": "note"}])
            with self.assertRaisesRegex(AnalysisInputError, "record_type"):
                load_records([temp], "development", "trial")


if __name__ == "__main__":
    unittest.main()
