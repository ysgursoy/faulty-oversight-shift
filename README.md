# Faulty Oversight Shift

A controlled in-context experiment on how systematically wrong evaluator feedback shifts an LLM's numeric answers, and whether that shift depends on task framing.

## Status

**Pipeline implemented; no model runs yet.** Data generation, the episode runner, classification, and analysis are implemented and validated offline with a mock model. No paid model run has been made, and the repository contains no empirical results.

## Research question

When an automated evaluator applies a systematic but incorrect rule, does repeated feedback move a model's numeric answers away from the stated protocol and toward the evaluator's value? Does that movement differ when identical arithmetic is framed as a neutral shipping task, an agricultural task, or a fictional human-dose task?

The experiment studies in-context adaptation inside one conversation. It does not train or fine-tune model weights.

## Experiment overview

The experiment uses synthetic cases and a deterministic numeric protocol. A model answers 40 cases per episode. Depending on condition, it receives no feedback, correct feedback, systematically faulty feedback, or faulty feedback with the evaluator's rule disclosed. The same numeric cases are paired across frames and conditions.

## Repository map

- [`docs/protocol.md`](docs/protocol.md): canonical study protocol.
- [`docs/reproduction.md`](docs/reproduction.md): reproduction contract.
- [`docs/data-card.md`](docs/data-card.md): synthetic data design and release policy.
- [`sources/`](sources/): references and source provenance.
- [`configs/`](configs/): versioned experiment settings.
- [`prompts/`](prompts/): exact model-visible English templates.
- [`data/README.md`](data/README.md): split roles and artifact policy.
- [`src/oversight_study/`](src/oversight_study/): the package (generation, runner, classification, analysis, validation).
- [`schemas/`](schemas/): JSON Schema generated from the typed record models.
- [`annotations/spec.yaml`](annotations/spec.yaml): classifier specification and manual-review procedure.
- [`tests/`](tests/): offline tests.

## Quick start

Python 3.11+ and [`uv`](https://docs.astral.sh/uv/) are required.

```bash
uv sync --frozen --all-extras
uv run oversight-study smoke
uv run python -m unittest discover -s tests
```

`smoke` runs the whole pipeline offline with a mock model and checks quotas, data separation, the request allowlist, the record contract, release validation, and recovery of a planted effect. Commands for reanalysis and model runs are in [`docs/reproduction.md`](docs/reproduction.md).

## Research-integrity controls

- Development and evaluation cases are procedurally separated.
- Model requests are built from an allowlist; case types, case ids, and condition names never reach the model.
- Raw outputs, retries, failures, and analysis provenance are retained.
- Paid model calls run only with an explicit flag; tests and CI never call an API.
- Negative and inconclusive results will be reported.
- In-context adaptation will not be described as training.

## License

Original code, documentation, and synthetic data in this repository are licensed under the [MIT License](LICENSE). Third-party sources retain their own licenses.
