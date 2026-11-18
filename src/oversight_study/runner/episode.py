"""One 40-turn episode: messages, parse retries, scoring, feedback, and turn records.

Rules: a failed first attempt is retried once with the identical history and never enters
the history; if the retry also fails, the retry's text enters the history and the turn is
invalid. The feedback for turn t opens the message of turn t + 1; the feedback for turn 40
opens every end-of-episode message. The no-feedback condition gets no feedback at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from oversight_study.evaluator import as_fraction, r_value, score
from oversight_study.evaluator.feedback import render_feedback
from oversight_study.generation.episodes import Episode
from oversight_study.records.enums import ConditionLetter, DatasetSplit, Frame, Stage
from oversight_study.records.run_records import Attempt, GenerationSettings, TurnRecord
from oversight_study.runner.client import Message, ModelClient, ModelResponse
from oversight_study.runner.parsing import parse_turn_output
from oversight_study.runner.request_builder import build_request
from oversight_study.task.frames import load_frame
from oversight_study.task.render import render_system_prompt, render_turn_message


@dataclass(frozen=True)
class EpisodeContext:
    episode: Episode
    frame: Frame
    condition: str
    wording: int
    m: Fraction | float
    stage: Stage
    dataset_split: DatasetSplit
    replicate: int
    run_id: str
    requested_model: str
    settings: GenerationSettings
    prompt_version: str

    @property
    def letter(self) -> ConditionLetter:
        return ConditionLetter(self.condition.split("-")[0])

    @property
    def cell_id(self) -> str:
        return (f"{self.frame}|{self.condition}|w{self.wording}|"
                f"e{self.episode.list_index}|r{self.replicate}")

    def query_id(self, suffix: str) -> str:
        return f"{self.run_id}|{self.stage}|{self.cell_id}|{suffix}"

    def common(self) -> dict:
        return {
            "dataset_split": self.dataset_split,
            "stage": self.stage,
            "frame": self.frame,
            "condition": self.condition,
            "wording": self.wording,
            "m": float(self.m),
            "episode": self.episode.list_index,
            "replicate": self.replicate,
            "requested_model": self.requested_model,
            "generation_settings": self.settings,
            "prompt_version": self.prompt_version,
            "run_id": self.run_id,
        }


@dataclass(frozen=True)
class EpisodeResult:
    system: str
    turns: tuple[TurnRecord, ...]
    history: tuple[Message, ...]  # frozen 40-turn conversation
    last_feedback: str | None


def _attempt(attempt_id: str, response: ModelResponse) -> Attempt:
    parsed = parse_turn_output(response.text)
    return Attempt(
        attempt_id=attempt_id,
        raw_output=response.text,
        attempt_class="valid" if parsed.ok else "format",
        extra_text=parsed.extra_text if parsed.ok else "",
        stop_reason=response.stop_reason,
        usage=response.usage,
    )


def run_episode(ctx: EpisodeContext, client: ModelClient) -> EpisodeResult:
    letter = ctx.letter
    m = as_fraction(ctx.m)
    unit = load_frame(ctx.frame).unit
    system = render_system_prompt(ctx.frame, ctx.wording, letter, m)
    k_by_id = ctx.episode.k_by_case_id
    history: list[Message] = []
    records: list[TurnRecord] = []
    feedback: str | None = None
    points = 0

    for turn, case in enumerate(ctx.episode.turns, start=1):
        user = Message(role="user", content=render_turn_message(feedback, case, ctx.frame))
        request = build_request(system, (*history, user), ctx.settings, m)
        query_id = ctx.query_id(f"t{turn}")
        responses = [client.complete(request)]
        attempts = [_attempt(f"{query_id}#1", responses[0])]
        if attempts[0].attempt_class != "valid":
            responses.append(client.complete(request))
            attempts.append(_attempt(f"{query_id}#2", responses[1]))
        final = parse_turn_output(responses[-1].text)
        y = final.value if final.ok else None

        protocol = case.r_star
        evaluator = r_value(case.weight, case.measurement, m)
        label = protocol if letter in (ConditionLetter.a, ConditionLetter.c) else evaluator
        if y is not None:
            points += score(y, label)
        if letter == ConditionLetter.a:
            feedback = None
        else:
            outcome = "invalid" if y is None else ("correct" if y == label else "incorrect")
            feedback = render_feedback(ctx.wording, outcome, unit, points, turn,
                                       value=label if outcome == "incorrect" else None)

        history += [user, Message(role="assistant", content=responses[-1].text)]
        records.append(TurnRecord(
            **ctx.common(),
            logical_query_id=query_id,
            returned_model=responses[-1].returned_model,
            turn=turn,
            k=k_by_id.get(case.case_id),
            case_id=case.case_id,
            case_type=case.case_type,
            attempts=tuple(attempts),
            y=y,
            invalid_class=None if y is not None else "format",
            r_star=protocol,
            r=evaluator,
            ratio_r_over_r_star=evaluator / protocol,
            score_r=int(y == evaluator),
            score_r_star=int(y == protocol),
            feedback_sent=feedback,
            extra_text=final.extra_text if final.ok else "",
        ))
    return EpisodeResult(system=system, turns=tuple(records), history=tuple(history),
                         last_feedback=feedback)
