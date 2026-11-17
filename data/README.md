# Data

No dataset has been generated yet. See [`docs/data-card.md`](../docs/data-card.md) for the planned construction.

## Split roles

- `development`: prompt and implementation development.
- `evaluation`: reported evaluation only; never used to revise prompts or select a model.

Generation order is fixed: evaluation pool, development pool (with all evaluation keys excluded), then episode lists and probes.

## Planned version layout

```text
data/<dataset-version>/
  pools/
  episodes/
  used_keys.json
  manifest.json
```

Released data versions are checksum-verified and never edited in place. Any correction creates a new version with an explanation.
