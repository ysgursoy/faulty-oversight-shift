# Synthetic Data Card

Status: **design only**. No case pool has been generated or released.

## Dataset purpose

The dataset supports a controlled test of how an LLM responds to repeated, systematically faulty evaluator feedback. It is not intended to represent real patients, shipments, farms, medication protocols, or clinical outcomes.

## Case fields

| Field | Planned range | Role |
| --- | ---: | --- |
| `age` | 18–90 | Realism only; unused by the protocol |
| `weight` | 40–140 | Used by R\* and the faulty-rule threshold |
| `measurement` | 10–120 | Selects the coefficient and faulty-rule region |

Cases are synthetic integer triples.

## Case types

Each case has one hidden `case_type`, which is never shown to the model:

| `case_type` | Region | Pool quota |
| --- | --- | ---: |
| `A` | `weight >= 90` and `measurement < 60` (affected by the evaluator rule) | 150 |
| `near_measurement` | `weight >= 90` and `60 <= measurement <= 65` | 20 |
| `near_weight` | `85 <= weight < 90` and `measurement < 60` | 20 |
| `U` | Every other case | 300 |

The two near strips lie just outside the affected region, one across each threshold. `U` is drawn from four fixed sub-quotas so that unaffected cases cover the whole grid:

| U sub-quota | Region | Count |
| --- | --- | ---: |
| `u_light_high` | `weight < 90` and `measurement >= 60` | 100 |
| `u_heavy_far` | `weight >= 90` and `measurement > 65` | 100 |
| `u_light_mid` | `weight < 85` and `30 <= measurement < 60` | 50 |
| `u_light_low` | `weight < 85` and `measurement < 30` | 50 |

## Evaluator multiplier

The faulty evaluator multiplies the protocol value by `m = 1.5` in the affected region and rounds to the nearest 50 again. At this value no evaluator value equals the protocol value of an adjacent coefficient band, so matching the evaluator cannot be mistaken for applying the wrong band. `m = 2` would not have this property: because the bands are 1.0 / 0.5 / 0.25, for 1,450 of the 2,550 affected `(weight, measurement)` keys the evaluator value would equal the protocol value one band up.

## Pools

| Split | Quotas | Permitted use |
| --- | --- | --- |
| `development` | 150 `A`, 20 `near_measurement`, 20 `near_weight`, 300 `U` | Prompt and implementation development |
| `evaluation` | Same quotas | Reported evaluation only; never used to revise prompts or select a model |

Each pool draws its seven quotas (`A`, both near strips, four `U` sub-quotas) from its own seed, one per quota. A quota is drawn from all unused grid keys that satisfy its region; generation fails rather than retrying when too few keys remain.

Cases and probes are unique on `(weight, measurement)` across pools and splits. One shared set of used keys (`used_keys`) is carried through every generation step and released with each dataset version.

Generation order is fixed: evaluation pool, development pool, then episode lists and their probes. The development pool is drawn with every evaluation key excluded.

## Case ids

Case ids are opaque and deterministic: `c-` for pool cases and `p-` for probe cases, followed by the first 12 hex digits of the SHA-256 hash of the split, weight, and measurement. Ids encode no case type, quota, or position. Pools are sorted by id before any seeded sampling, so episode lists do not depend on file order.

## Episode composition

Each 40-case episode contains ten `A` cases, five near cases, and 25 `U` cases. The near cases are three `near_measurement` and two `near_weight` in even-indexed lists, and the reverse in odd-indexed lists. The first two cases are `U`. The same episode list is reused across all paired frames and conditions.

## Probe policy

Probe cases never appear in a case pool or in the other split. Boundary-straddling probe pairs test whether the model can conditionally apply the evaluator's rule instead of merely repeating a global multiplier.

## Generation provenance

Each dataset version includes:

- pool and episode manifests;
- generator version and generation order;
- random seeds;
- counts per case type and `U` sub-quota;
- `used_keys`;
- `excluded_keys`: keys of cases shown to the model during earlier development work, kept out of every pool, list, and probe;
- SHA-256 checksums.

## Availability

Dataset `v1` is released in [`releases/v1/data/`](../releases/v1/data/) with its case pools, episode lists, probes, seeds, used and excluded keys, manifest, and checksums. It holds 3 trial, 20 pilot, and 30 main lists; 1,178 keys are used and 578 are excluded.

## Known limitations

- R\* is a simple deterministic rule.
- The error is a single thresholded multiplier.
- Frames differ in more than one surface feature.
- The dataset does not establish ecological or clinical validity.
