"""Papiti v0.5: training-only synthetic arithmetic curriculum.

The target of each arithmetic example is generated before training from the
arithmetic operation, to supply supervised examples. At *inference*, there is
no arithmetic function, regex classifier, prompt lookup, calculator or canned
answer: only the trained autoregressive Transformer generates the reply.

A disjoint set of operand pairs is reserved for honest generalization tests,
including previously seen user prompts. The arithmetic labels are *not*
looked up in inference, and the heldout pairs never enter tokenizer training.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from papiti_lm.corpus import Example


# Probe questions are evaluation inputs, not an inference routing table.
# Reserve the unordered operand pairs and their reversed forms from training.
PROBE_PAIRS: tuple[tuple[int, int], ...] = (
    (1, 1), (23, 47), (8, 9), (39, 56), (44, 44),
    (2, 7), (12, 32), (58, 29),
)


@dataclass(frozen=True)
class MathSplit:
    train: list[Example]
    valid: list[Example]
    train_pairs: frozenset[tuple[int, int]]
    valid_pairs: frozenset[tuple[int, int]]


def _pair(a: int, b: int) -> tuple[int, int]:
    return min(a, b), max(a, b)


def _example(a: int, b: int, rng: random.Random, group: str, *, probe: bool = False) -> Example:
    if probe:
        question = f"{a}+{b}은?"
    else:
        style = rng.randrange(10)
        if style < 8:
            question = f"{a}+{b}은?"
        elif style == 8:
            question = f"{a} + {b} = ?"
        else:
            question = f"{a} 더하기 {b}는?"
    # This is an explicit training label, *never* an inference implementation.
    return Example(group=group, user=question, assistant=str(a + b))


def arithmetic_examples(
    seed: int = 42,
    train_count: int = 3000,
    valid_count: int = 160,
) -> MathSplit:
    """Generate train/eval arithmetic examples with pair-level disjointness.

    Only addition of nonnegative integers from 0..99 is trained. The model
    is not expected to generalize to subtraction, large integers, or other
    mathematical operations.
    """
    if not 1 <= train_count <= 4500 or not 1 <= valid_count <= 500:
        raise ValueError("arithmetic example counts outside supported range")
    reserved = frozenset(_pair(a, b) for a, b in PROBE_PAIRS)
    pool = [
        (a, b) for a in range(100) for b in range(a, 100)
        if (a, b) not in reserved
    ]
    if train_count + valid_count > len(pool):
        raise ValueError("arithmetic train + valid counts exceed available pairs")
    rng = random.Random(seed)
    rng.shuffle(pool)
    train_pairs = pool[:train_count]
    valid_pairs = pool[train_count:train_count + valid_count]
    train = [_example(a, b, rng, "arithmetic-train") for a, b in train_pairs]
    valid = [_example(a, b, rng, "arithmetic-heldout") for a, b in valid_pairs]
    valid += [_example(a, b, rng, "arithmetic-probe", probe=True)
              for a, b in PROBE_PAIRS]
    train_set = frozenset(train_pairs)
    valid_set = frozenset(valid_pairs) | reserved
    if train_set & valid_set:
        raise AssertionError("arithmetic train/validation operand leakage")
    return MathSplit(train, valid, train_set, valid_set)


def augment_language(examples: list[Example]) -> list[Example]:
    """Safe, training-only punctuation variants, using only training groups.

    This is ordinary data augmentation: it DOES NOT add a runtime rule for
    matching a specific input to a specific response.
    """
    augmented: list[Example] = []
    for example in examples:
        original = example.user.strip()
        if original.endswith("!"):
            alternative = original.rstrip("!")
        elif original.endswith("?"):
            alternative = original.rstrip("?")
        else:
            alternative = original + "!"
        if alternative and alternative != original and len(alternative) <= 256:
            augmented.append(Example(example.group, alternative, example.assistant))
    return augmented
