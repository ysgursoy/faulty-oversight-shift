"""Parse model output: the per-turn JSON answer and end-of-episode numeric answers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

_DECODER = json.JSONDecoder()
_NUMBER = re.compile(r"-?\d{1,3}(?:,\d{3})+(?!\d)|-?\d+")
_LAST_LINE = re.compile(r"\**(-?\d{1,3}(?:,\d{3})+|-?\d+)\**\.?")


@dataclass(frozen=True)
class ParseResult:
    ok: bool
    value: int | None
    extra_text: str  # text outside the accepted JSON object; whole text when invalid


def parse_turn_output(text: str) -> ParseResult:
    """Accept the first JSON object whose `value` is an integer.

    Surrounding text is allowed and returned as `extra_text`. Booleans, floats
    (including `1500.0`), and numeric strings are not integers.
    """
    position = text.find("{")
    while position != -1:
        try:
            obj, end = _DECODER.raw_decode(text, position)
        except json.JSONDecodeError:
            position = text.find("{", position + 1)
            continue
        if isinstance(obj, dict) and "value" in obj and type(obj["value"]) is int:
            extra = (text[:position] + text[end:]).strip()
            return ParseResult(ok=True, value=obj["value"], extra_text=extra)
        position = text.find("{", position + 1)
    return ParseResult(ok=False, value=None, extra_text=text.strip())


def parse_number_answer(text: str) -> int | None:
    """Return the numeric answer, or None when it cannot be read unambiguously.

    If the last non-empty line consists of a single integer (thousands separators
    and a trailing period allowed), that integer is the answer, whatever working
    precedes it. Otherwise the whole text must contain exactly one distinct integer;
    any second distinct number, including the fractional part of a decimal, makes
    the answer ambiguous.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        last = _LAST_LINE.fullmatch(lines[-1])
        if last:
            return int(last.group(1).replace(",", ""))
    numbers = {int(match.replace(",", "")) for match in _NUMBER.findall(text)}
    if len(numbers) != 1:
        return None
    return numbers.pop()
