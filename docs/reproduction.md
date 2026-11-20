# Reproduction Guide

Status: **design version 1, completed.** The released records are in [`releases/v1/`](../releases/v1/).

## 1. Environment

- Python 3.11 or newer
- [`uv`](https://docs.astral.sh/uv/)

```bash
uv sync --frozen --all-extras
```

## 2. Offline validation

```bash
uv run oversight-study smoke
uv run python -m unittest discover -s tests
```

`smoke` generates a small dataset, runs a mock-model trial over every frame, condition, and wording, classifies the answers with a mock classifier, seals and validates the run, and checks that the analysis recovers a planted effect, returns about zero without one, and stops on malformed input. It makes no API call.

## 3. Reanalysis of a released record set

A release directory holds one dataset (`data/`) and sealed runs (`runs/<run-id>/`). The released set is `releases/v1`; it also keeps the pre-check records under `prechecks/`.

```bash
uv run oversight-study validate-release releases/v1
uv run oversight-study analyze releases/v1 --stage main --split evaluation --out results
```

`analyze` writes tables, figures, and a manifest with input run ids, parameters, and checksums under `results/<analysis-id>/`. Rerunning it on the same release reproduces the tables exactly; the released analysis is [`results/main-evaluation-r0-c0557ee2/`](../results/main-evaluation-r0-c0557ee2/). Classifier outputs and the manual review for the main run are in [`annotations/main-20261119T203514Z-2ad2bde3/`](../annotations/main-20261119T203514Z-2ad2bde3/); they are not covered by the run's seal and carry their own `checksums.sha256`.

## 4. Running the pipeline

The same commands run offline with `--mock <policy>` and against the model API with `--allow-paid` (which also needs `ANTHROPIC_API_KEY`). Without either flag, model commands refuse to run.

```bash
uv run oversight-study generate --dataset-version <version> --out data/<version>
uv run oversight-study precheck --dataset data/<version> --split development --out <dir> --mock protocol
uv run oversight-study run --config configs/stages/trial.yaml --dataset data/<version> --mock evaluator
uv run oversight-study classify --run <run-dir> --dataset data/<version> --mock
uv run oversight-study seal <run-dir>
```

`run --workers N` runs up to N episodes at once; each episode's turns stay sequential, episodes start in the stage's fixed shuffled order, and records are identical to a sequential run apart from file order and timestamps.

Development data are used for the pre-check, the trial, and the pilot. Evaluation data are used only after the protocol, prompts, model settings, sample size, and analysis plan are fixed and recorded; model commands on evaluation data refuse to run otherwise. A paid rerun produces new observations, which may differ from released ones even with identical seeds and settings.

## 5. Notebook contract

Released notebooks may provide exploration and reader-facing walkthroughs, but they import canonical analysis functions from the package. No reported metric or statistical test exists only inside a notebook. Released notebooks run top to bottom in a clean kernel against a named release or fixture and disclose their input artifact ids.
