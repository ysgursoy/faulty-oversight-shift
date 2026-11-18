"""End-of-episode questions, each asked on its own copy of the frozen 40-turn history.

No question sees another's answer. Feedback episodes get 12 questions (rule, three value,
four probe-protocol, four probe-evaluator); the no-feedback condition gets only the three
value questions. Numeric answers must contain exactly one integer; there is no retry.
"""

from __future__ import annotations

from oversight_study.evaluator import as_fraction
from oversight_study.records.enums import ConditionLetter, QuestionType
from oversight_study.records.run_records import EpisodeEndRecord
from oversight_study.runner.client import Message, ModelClient
from oversight_study.runner.episode import EpisodeContext, EpisodeResult
from oversight_study.runner.parsing import parse_number_answer
from oversight_study.runner.request_builder import build_request
from oversight_study.task.render import render_episode_end_message


def planned_questions(ctx: EpisodeContext) -> list[tuple[QuestionType, object]]:
    """(question, case) in asking order; case is None for the rule question."""
    episode = ctx.episode
    values = [(QuestionType.value, case) for case in episode.last_affected]
    if ctx.letter == ConditionLetter.a:
        return values
    probes = list(episode.probes)
    return ([(QuestionType.rule, None)] + values
            + [(QuestionType.probe_protocol, p) for p in probes]
            + [(QuestionType.probe_evaluator, p) for p in probes])


def ask_episode_end(result: EpisodeResult, ctx: EpisodeContext,
                    client: ModelClient) -> list[EpisodeEndRecord]:
    frozen: tuple[Message, ...] = result.history
    feedback = result.last_feedback if ctx.letter != ConditionLetter.a else None
    m = as_fraction(ctx.m)
    records = []
    for question, case in planned_questions(ctx):
        text = render_episode_end_message(feedback, question, ctx.frame, case)
        request = build_request(result.system, (*frozen, Message(role="user", content=text)),
                                ctx.settings, m)
        response = client.complete(request)
        case_id = getattr(case, "case_id", None)
        if question == QuestionType.rule:
            y, invalid = None, None
        else:
            y = parse_number_answer(response.text)
            invalid = None if y is not None else "format"
        records.append(EpisodeEndRecord(
            **ctx.common(),
            logical_query_id=ctx.query_id(f"end|{question}|{case_id or 'rule'}"),
            returned_model=response.returned_model,
            question_type=question,
            case_id=case_id,
            pair=getattr(case, "pair", None) if question.value.startswith("probe") else None,
            side=getattr(case, "side", None) if question.value.startswith("probe") else None,
            raw_answer=response.text,
            y=y,
            invalid_class=invalid,
        ))
    return records
