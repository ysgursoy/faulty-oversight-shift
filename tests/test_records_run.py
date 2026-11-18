import unittest

from pydantic import ValidationError

from oversight_study.records.run_records import EpisodeEndRecord, TurnRecord

USAGE = {"input_tokens": 100, "output_tokens": 5}
VALID = {"attempt_id": "q#1", "raw_output": '{"value": 1500}', "attempt_class": "valid",
         "usage": USAGE}
FORMAT = {"attempt_id": "q#1", "raw_output": "oops", "attempt_class": "format", "usage": USAGE}
BASE = {
    "dataset_split": "development", "stage": "trial", "frame": "human", "condition": "b-1.5",
    "wording": 1, "m": 1.5, "episode": 0, "requested_model": "model-x",
    "returned_model": "model-x", "generation_settings": {"max_tokens": 64},
    "prompt_version": "0" * 64, "run_id": "run", "logical_query_id": "run|cell|t1",
}
TURN = {**BASE, "turn": 5, "k": 1, "case_id": "c-1", "case_type": "A", "attempts": (VALID,),
        "y": 1500, "r_star": 1000, "r": 1500, "ratio_r_over_r_star": 1.5, "score_r": 1,
        "score_r_star": 0, "feedback_sent": "Evaluator result: CORRECT (1 point). Total: 1/5"}
END = {**BASE, "question_type": "probe_evaluator", "case_id": "p-1", "pair": "weight",
       "side": "inside", "raw_answer": "1500", "y": 1500}


def bad_turn(**changes):
    return {**TURN, **changes}


class TurnRecordTests(unittest.TestCase):
    def test_valid(self) -> None:
        self.assertEqual(TurnRecord(**TURN).condition_letter, "b")

    def test_k_only_for_affected(self) -> None:
        TurnRecord(**bad_turn(case_type="U", k=None, r=1000, ratio_r_over_r_star=1.0,
                              score_r=0, score_r_star=0, y=1200))
        for changes in ({"k": None}, {"case_type": "U"}, {"k": 11}):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                TurnRecord(**bad_turn(**changes))

    def test_y_and_invalid_class(self) -> None:
        invalid = bad_turn(y=None, invalid_class="format", attempts=(FORMAT, FORMAT),
                           score_r=0, score_r_star=0)
        TurnRecord(**invalid)
        with self.assertRaises(ValidationError):
            TurnRecord(**{**invalid, "invalid_class": None})
        with self.assertRaises(ValidationError):
            TurnRecord(**bad_turn(invalid_class="refusal"))

    def test_y_never_bool(self) -> None:
        with self.assertRaises(ValidationError):
            TurnRecord(**bad_turn(y=True))

    def test_ratio_and_scores(self) -> None:
        with self.assertRaises(ValidationError):
            TurnRecord(**bad_turn(ratio_r_over_r_star=1.4))
        with self.assertRaises(ValidationError):
            TurnRecord(**bad_turn(score_r=0))
        with self.assertRaises(ValidationError):
            TurnRecord(**bad_turn(score_r_star=2))

    def test_attempt_rules(self) -> None:
        TurnRecord(**bad_turn(attempts=(FORMAT, VALID)))
        for attempts in ((), (VALID, VALID), (FORMAT,), (FORMAT, FORMAT, VALID)):
            with self.subTest(n=len(attempts)), self.assertRaises(ValidationError):
                TurnRecord(**bad_turn(attempts=attempts))

    def test_feedback_presence_follows_condition(self) -> None:
        TurnRecord(**bad_turn(condition="a", feedback_sent=None))
        with self.assertRaises(ValidationError):
            TurnRecord(**bad_turn(condition="a"))
        with self.assertRaises(ValidationError):
            TurnRecord(**bad_turn(feedback_sent=None))

    def test_unknown_values(self) -> None:
        for changes in ({"condition": "b"}, {"frame": "finance"}, {"stage": "stage-x"},
                        {"dataset_split": "holdout"}, {"prompt_version": "abc"}, {"turn": 41}):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                TurnRecord(**bad_turn(**changes))


class EpisodeEndRecordTests(unittest.TestCase):
    def test_valid_probe_value_and_rule(self) -> None:
        EpisodeEndRecord(**END)
        EpisodeEndRecord(**{**END, "question_type": "value", "pair": None, "side": None,
                            "case_id": "c-9"})
        EpisodeEndRecord(**{**END, "question_type": "rule", "pair": None, "side": None,
                            "case_id": None, "y": None, "raw_answer": "none"})

    def test_pair_side_only_for_probes(self) -> None:
        with self.assertRaises(ValidationError):
            EpisodeEndRecord(**{**END, "pair": None})
        with self.assertRaises(ValidationError):
            EpisodeEndRecord(**{**END, "question_type": "value", "case_id": "c-9"})

    def test_case_id_none_only_for_rule(self) -> None:
        with self.assertRaises(ValidationError):
            EpisodeEndRecord(**{**END, "case_id": None})
        with self.assertRaises(ValidationError):
            EpisodeEndRecord(**{**END, "question_type": "rule", "pair": None, "side": None,
                                "y": None})

    def test_numeric_answers_follow_y_rule(self) -> None:
        EpisodeEndRecord(**{**END, "y": None, "invalid_class": "refusal"})
        with self.assertRaises(ValidationError):
            EpisodeEndRecord(**{**END, "y": None})
        with self.assertRaises(ValidationError):
            EpisodeEndRecord(**{**END, "y": False})

    def test_no_feedback_condition_asks_only_values(self) -> None:
        with self.assertRaises(ValidationError):
            EpisodeEndRecord(**{**END, "condition": "a"})
        EpisodeEndRecord(**{**END, "condition": "a", "question_type": "value", "pair": None,
                            "side": None, "case_id": "c-9"})


if __name__ == "__main__":
    unittest.main()
