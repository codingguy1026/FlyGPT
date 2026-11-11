"""Explicit, consented JSONL training data and character tokenization.

Each record: {"group": "...", "user": "...", "assistant": "..."}.
A group is a topical/source cluster kept entirely on one side of the split.
No user conversations, messages or private files are collected automatically.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path


PAD, BOS, USER, ASSISTANT, EOS, UNK = range(6)
RESERVED = 6


@dataclass(frozen=True)
class Example:
    group: str
    user: str
    assistant: str


def load_examples(path: str | Path) -> list[Example]:
    records: list[Example] = []
    unique: set[tuple[str, str]] = set()
    with Path(path).open("r", encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL record at line {number}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"line {number}: expected an object")
            fields = (item.get("group"), item.get("user"), item.get("assistant"))
            if not all(isinstance(x, str) and x.strip() for x in fields):
                raise ValueError(f"line {number}: nonempty group/user/assistant strings required")
            group, user, assistant = (x.strip() for x in fields)
            if len(group) > 80 or len(user) > 256 or len(assistant) > 256:
                raise ValueError(f"line {number}: record exceeds length limits")
            identity = (user.casefold(), assistant.casefold())
            if identity in unique:
                raise ValueError(f"line {number}: duplicate dialogue")
            unique.add(identity)
            records.append(Example(group, user, assistant))
    if len(records) < 10 or len({x.group for x in records}) < 3:
        raise ValueError("at least 10 examples spanning 3 different groups are required")
    return records


def split_by_group(examples: list[Example], seed: int = 42, fraction: float = 0.2):
    """Hold out whole groups so memorized pairs do not inflate validation."""
    if not 0 < fraction < 0.5:
        raise ValueError("fraction must be between 0 and 0.5")
    groups = sorted({item.group for item in examples})
    if len(groups) < 3:
        raise ValueError("need at least three groups")
    random.Random(seed).shuffle(groups)
    count = min(len(groups) - 1, max(1, round(len(groups) * fraction)))
    held_out = set(groups[:count])
    train = [x for x in examples if x.group not in held_out]
    valid = [x for x in examples if x.group in held_out]
    return train, valid


class CharTokenizer:
    def __init__(self, characters: list[str]):
        if any(len(x) != 1 for x in characters) or len(set(characters)) != len(characters):
            raise ValueError("characters must be unique, individual Unicode code points")
        self.characters = sorted(characters)
        self.char_to_id = {c: i + RESERVED for i, c in enumerate(self.characters)}

    @classmethod
    def from_examples(cls, examples: list[Example]):
        characters = set()
        for item in examples:
            characters.update(item.user)
            characters.update(item.assistant)
        return cls(sorted(characters))

    @property
    def vocab_size(self) -> int:
        return RESERVED + len(self.characters)

    def encode(self, text: str) -> list[int]:
        return [self.char_to_id.get(c, UNK) for c in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(
            self.characters[token - RESERVED] if RESERVED <= token < self.vocab_size
            else "�" if token == UNK else ""
            for token in tokens
        )

    def prompt(self, user_text: str) -> list[int]:
        return [BOS, USER] + self.encode(user_text) + [ASSISTANT]

    def sequence(self, item: Example, context: int) -> tuple[list[int], list[int]]:
        prefix = self.prompt(item.user)
        ids = prefix + self.encode(item.assistant) + [EOS]
        if len(ids) - 1 > context:
            raise ValueError(
                f"example in group {item.group!r} exceeds context {context}; "
                "increase --context or shorten the record"
            )
        # Predict answer characters and EOS only; never optimize copying the prompt.
        input_ids, labels = ids[:-1], ids[1:]
        labels[:len(prefix) - 1] = [-100] * (len(prefix) - 1)
        return input_ids, labels
