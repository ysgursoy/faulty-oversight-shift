# Data

The released dataset is [`releases/v1/data/`](../releases/v1/data/). See [`docs/data-card.md`](../docs/data-card.md) for its construction.

## Split roles

- `development`: prompt and implementation development (pre-check, trial, pilot).
- `evaluation`: reported evaluation only; never used to revise prompts or select a model.

Generation order is fixed: evaluation pool, development pool (with all evaluation keys excluded), then episode lists and probes. Keys listed in `excluded_keys.json` are treated as already used.

## Version layout

```text
releases/<dataset-version>/data/
  pools/
  episodes/
  used_keys.json
  excluded_keys.json
  manifest.json
```

Released data versions are checksum-verified and never edited in place. Any correction creates a new version with an explanation.
