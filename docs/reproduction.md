# Reproduction Guide

Status: **design only**. The experiment runner and analysis pipeline are not yet published, and no results are available. Environment requirements and setup commands will be added with the runnable implementation.

## 1. Reproduction contract

The released implementation will distinguish three activities:

- reanalysis of released records, which must regenerate every reported table and figure from validated manifests and checksums;
- an offline mock run that validates generation, feedback, parsing, retries, schemas, and analysis recovery without an API call;
- an explicit Anthropic API rerun, which produces new observations and may differ despite identical seeds and temperature.

Automated checks run offline on fixtures and require no API credential.

## 2. Notebook contract

Released notebooks may provide exploration and reader-facing walkthroughs, but they import canonical analysis functions from the package. No reported metric or statistical test exists only inside a notebook. Released notebooks run top to bottom in a clean kernel against a named release or fixture and disclose their input artifact ids.
