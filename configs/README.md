# Configuration

Versioned settings used by runs. Configuration files contain no credentials. Every run stores its fully resolved configuration and hashes the source configuration, prompts, dataset, dependency lock, and code commit.

| File | Contents |
| --- | --- |
| `models/claude.yaml` | Provider, dated model id, sampling and reasoning settings, token limit, caching, transport retries, price assumptions. `model_id: TBD` until a model version is fixed; paid runs refuse it. |
| `stages/trial.yaml` | Three development lists in every frame, condition, and wording; mechanical checks only. |
| `stages/pilot.yaml` | Development pilot: core cells, a repeat of the first five lists, optional cells. |
| `stages/main.yaml` | Main experiment on the evaluation split. `list_count` is fixed after the pilot. |
| `precheck.yaml` | Single-turn protocol check: frames, wordings, threshold, and case quotas. |
| `analysis/primary.yaml` | Equivalence margin, smallest effect of interest, list cap, pilot size, bootstrap settings. Undecided values are `null` and block the steps that need them. |
| `data/seeds-dev.yaml` | Development seed bases. Evaluation seeds are not versioned before release. |

Models live in `src/oversight_study/config.py`.
