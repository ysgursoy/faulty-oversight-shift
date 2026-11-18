"""Request payload allowlist check: hidden metadata must never reach the model.

Forbidden in any model-visible text: case ids, case-type names, condition ids, the
words `affected` / `case_type`, a `"k"` field, and the multiplier itself — except inside
the disclosed-rule sentence of the system prompt in condition `f`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from fractions import Fraction

from oversight_study.runner.client import ModelRequest
from oversight_study.task.templates import load_template

PATTERNS = {
    "case id": re.compile(r"\b[cp]-[0-9a-f]{12}\b"),
    "case type": re.compile(r"\b(?:near_measurement|near_weight|case[ _]type)\b", re.IGNORECASE),
    "condition id": re.compile(r"\b[bf]-\d+(?:\.\d+)?\b"),
    "affected flag": re.compile(r"\baffected\b", re.IGNORECASE),
    "k field": re.compile(r'"k"\s*:'),
}


class PayloadLeakError(ValueError):
    """A request would show hidden metadata to the model."""


def _explicit_rule_pattern() -> re.Pattern:
    template = re.escape(load_template("system/explicit_rule.txt"))
    for field in ("weight_label", "measurement_label", "m"):
        template = template.replace(re.escape("{" + field + "}"), ".+?")
    return re.compile(template)


def check_payload(request: ModelRequest, m: Fraction | float,
                  extra_forbidden: Iterable[str] = ()) -> None:
    """Raise PayloadLeakError naming every leak found in the request text."""
    system = _explicit_rule_pattern().sub("", request.system)
    texts = [system, *(message.content for message in request.messages)]
    leaks = []
    for name, pattern in PATTERNS.items():
        for text in texts:
            match = pattern.search(text)
            if match:
                leaks.append(f"{name}: {match.group(0)!r}")
                break
    m_text = f"{float(m):g}"
    if "." in m_text:  # an integer m cannot be told apart from ordinary numbers
        m_pattern = re.compile(rf"(?<![\d.]){re.escape(m_text)}(?![\d])")
        if any(m_pattern.search(text) for text in texts):
            leaks.append(f"multiplier: {m_text!r}")
    for token in extra_forbidden:
        if any(token in text for text in texts):
            leaks.append(f"forbidden token: {token!r}")
    if leaks:
        raise PayloadLeakError("; ".join(leaks))
