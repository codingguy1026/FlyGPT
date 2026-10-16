"""Deterministic, approximate paraphrase-grouped train/val/test splits.

Mirrors the same seeded grouping recipe used in FlyGPT's earlier 93.16%
256-neuron learning-curve experiment. Importantly, examples are clustered
*before* split assignment; neither validation nor test is used for training.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import random
import re


def shingles(text: str) -> set[str]:
    normalized = re.sub(r"\s+", "", text.strip().casefold())
    if len(normalized) <= 3:
        return {normalized}
    return {normalized[i:i+3] for i in range(len(normalized) - 2)}


def group_near_duplicates(examples, threshold: float = 0.86):
    n = len(examples)
    parent = list(range(n))
    grams = [shingles(x.text) for x in examples]

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for i in range(n):
        a = grams[i]
        for j in range(i + 1, n):
            b = grams[j]
            if min(len(a), len(b)) == 0:
                continue
            if min(len(a), len(b)) / max(len(a), len(b)) < threshold:
                continue
            overlap = len(a & b) / (len(a) + len(b) - len(a & b))
            if overlap >= threshold:
                union(i, j)
    clusters = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(i)
    return list(clusters.values())


def grouped_three_way_split(examples, seed: int):
    groups = group_near_duplicates(examples)
    by_route = defaultdict(list)
    for group in groups:
        major = sorted(Counter(examples[i].route for i in group).items(),
                       key=lambda item: (-item[1], item[0]))[0][0]
        by_route[major].append(group)
    rng = random.Random(seed)
    train_ids, val_ids, test_ids = [], [], []
    for route in sorted(by_route):
        clusters = by_route[route][:]
        rng.shuffle(clusters)
        if len(clusters) < 4:
            raise ValueError(f"Insufficient independent {route} clusters")
        n_test = max(1, round(len(clusters) * .15))
        n_val = max(1, round(len(clusters) * .15))
        test_ids.extend(i for g in clusters[:n_test] for i in g)
        val_ids.extend(i for g in clusters[n_test:n_test + n_val] for i in g)
        train_ids.extend(i for g in clusters[n_test + n_val:] for i in g)

    a, b, c = set(train_ids), set(val_ids), set(test_ids)
    if a & b or a & c or b & c or a | b | c != set(range(len(examples))):
        raise AssertionError("Overlapping/missing example in group split")
    known = {row.route for row in examples}
    if any({examples[i].route for i in part} != known
           for part in (train_ids, val_ids, test_ids)):
        raise ValueError("A split has missing route labels")
    return (
        [examples[i] for i in train_ids],
        [examples[i] for i in val_ids],
        [examples[i] for i in test_ids],
        {"n_clusters": len(groups),
         "largest_cluster": max(len(g) for g in groups),
         "train": len(train_ids), "validation": len(val_ids), "test": len(test_ids),
         "test_class_counts": dict(Counter(examples[i].route for i in test_ids))},
    )
