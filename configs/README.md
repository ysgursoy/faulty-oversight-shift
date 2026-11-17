# Configuration

No runnable experiment configuration is included yet. This directory will contain the exact versioned configurations used by released runs, including the dated Anthropic model id, sampling and reasoning settings, limits, retry behavior, estimands, bootstrap settings, missing-data handling, and sample-size rules.

Configuration files contain no credentials. Every run stores its fully resolved configuration and hashes the source configuration, prompts, dataset, dependency lock, and code commit.

Each configuration identifies the protocol version it implements.
