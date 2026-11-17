# Faulty Oversight Shift

A controlled in-context experiment on how systematically wrong evaluator feedback shifts an LLM's numeric answers, and whether that shift depends on task framing.

## Status

**Design only.** The experiment has not been implemented or run, and this repository contains no empirical results yet.

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

The Python package, offline tests, and setup instructions will be published with the runnable implementation.

## Research-integrity controls

- Development and evaluation cases are procedurally separated.
- Raw outputs, retries, failures, and analysis provenance will be retained.
- Negative and inconclusive results will be reported.
- In-context adaptation will not be described as training.

## License

Original code and documentation in this repository are licensed under the [MIT License](LICENSE). Third-party sources retain their own licenses.
