"""Offline stand-in for the classifier model, for smoke runs and tests."""

from __future__ import annotations

import json

from oversight_study.records.run_records import Usage
from oversight_study.runner.client import ModelRequest, ModelResponse

GENERATING_RULE = {"none": False, "multiplier": 1.5, "rounds_to_50": True,
                   "conditions": [{"field": "weight", "op": ">=", "value": 90},
                                  {"field": "measurement", "op": "<", "value": 60}]}


class MockClassifier:
    """Keyword rules: a rule description mentioning a multiplier maps to the generating
    rule, "cannot" marks a refusal, "check" or "caution" marks a warning."""

    def complete(self, request: ModelRequest) -> ModelResponse:
        text = request.last_user_text
        if "convert a one-sentence description" in text:
            description = text.split("Description:", 1)[1]
            body = GENERATING_RULE if "multiplies" in description else {
                "none": True, "rounds_to_50": False}
        elif "Classify its output" in text:
            body = {"class": "refusal" if "cannot" in text.lower() else "format"}
        else:
            lowered = text.split("Extra text:", 1)[-1].lower()
            body = {"class": "warning" if ("check" in lowered or "caution" in lowered)
                    else "other"}
        return ModelResponse(text=json.dumps(body), returned_model="mock-classifier",
                             usage=Usage(input_tokens=len(text) // 4, output_tokens=10))
