# Prompts

Exact English model-visible templates used by the experiment. Templates use Python `str.format` fields; literal braces are doubled.

Every prompt used in a run is versioned and hashed. Templates must expose only allowlisted model-visible fields. Case type, affected-case index, implementation ids, condition name, and multiplier must not leak except through permitted feedback or the explicit-rule condition.

## Layout

- `system/wording_1.txt`, `system/wording_2.txt`: system prompts for the two wordings. Frame fields: `role`, `object`, `age_label`, `weight_label`, `measurement_label`, `output_label`, `unit`.
- `system/evaluator_sentence_{1,2}.txt`: the sentence announcing evaluator feedback. It fills the `{evaluator}` field. In the no-feedback condition the field is empty and the line or sentence it occupied is removed; the two-stage structure stays.
- `system/explicit_rule.txt`: the disclosed evaluator error, appended directly after the evaluator sentence in the disclosed-rule condition. `{m}` is rendered in compact form (`1.5`, `2`).
