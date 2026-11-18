"""Dataset-level separation: pools, splits, and probes never share a key."""

from __future__ import annotations

from oversight_study.generation.dataset import LIST_SPLIT, Dataset, check_disjoint
from oversight_study.validation.episodes import check_episode


def check_split_overlap(dataset: Dataset) -> None:
    """Keys are disjoint across pools and splits; probes repeat only within one split's lists.

    Also checks every episode list's composition against its split's pool.
    """
    check_disjoint(dataset)
    for use, episodes in dataset.episodes.items():
        pools = dataset.pools[LIST_SPLIT[use]]
        for index, episode in enumerate(episodes):
            if episode.list_index != index:
                raise ValueError(f"{use} lists out of order at position {index}")
            check_episode(episode, pools)
