"""Explicit, consented JSONL training data and character tokenization.

Each record: {"group": "...", "user": "...", "assistant": "..."}.
A group is a topical/source cluster kept entirely on one side of the split.
No user conversations, messages or private files are collected automatically.
"""
from __future__ import annotations

import json
import random
import re
import unicodedata
from collections import Counter
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


class JamoTokenizer(CharTokenizer):
    """Decompose precomposed Korean syllables into Unicode NFD jamo tokens.

    The vocabulary is still derived ONLY from the training split. In contrast
    to syllable tokenization, a novel valid Hangul syllable can be encoded when
    its constituent jamo were observed in training. This is not a pretrained
    tokenizer and it does not guarantee valid Korean generation.
    """

    @classmethod
    def from_examples(cls, examples: list[Example]):
        characters: set[str] = set()
        for item in examples:
            characters.update(unicodedata.normalize("NFD", item.user))
            characters.update(unicodedata.normalize("NFD", item.assistant))
        return cls(sorted(characters))

    def encode(self, text: str) -> list[int]:
        return super().encode(unicodedata.normalize("NFD", text))

    def decode(self, tokens: list[int]) -> str:
        return unicodedata.normalize("NFC", super().decode(tokens))


def _bpe_chunks(value: str) -> list[list[str]]:
    """Split words and whitespace; never learn merges across word boundaries."""
    return [list(part) for part in re.findall(r"\s+|\S+", value)]


def _merge_pair(chunks: list[str], pair: tuple[str, str]) -> list[str]:
    merged: list[str] = []
    index = 0
    while index < len(chunks):
        if index + 1 < len(chunks) and (chunks[index], chunks[index + 1]) == pair:
            merged.append(chunks[index] + chunks[index + 1])
            index += 2
        else:
            merged.append(chunks[index])
            index += 1
    return merged


class SubwordTokenizer(CharTokenizer):
    """Small, train-only BPE over Unicode NFC Korean syllables and punctuation.

    It never emits a standalone decomposed Hangul jamo token unless a jamo was
    explicitly present in training text. Unknown codepoints still use UNK.
    Learned whole-character pieces make Korean spacing and short recurring
    phrases available as units rather than forcing every syllable to be
    assembled from component jamo.
    """

    def __init__(self, tokens: list[str], merges: list[tuple[str, str]]):
        if any(not token or len(token) > 12 for token in tokens):
            raise ValueError("invalid BPE token")
        if len(set(tokens)) != len(tokens):
            raise ValueError("duplicate BPE token")
        self.characters = sorted(tokens)  # Consistent with checkpoint format v1.
        self.char_to_id = {token: i + RESERVED for i, token in enumerate(self.characters)}
        self.merges = [tuple(pair) for pair in merges]
        if any(len(pair) != 2 or pair[0] + pair[1] not in self.char_to_id
               for pair in self.merges):
            raise ValueError("invalid BPE merges")

    @classmethod
    def from_examples(cls, examples: list[Example], max_merges: int = 256):
        if not 0 <= max_merges <= 1024:
            raise ValueError("max_merges must be 0..1024")
        texts = [text for row in examples for text in (row.user, row.assistant)]
        chunks = [part for text in texts for part in _bpe_chunks(text)]
        base = set(char for text in texts for char in text)
        merges: list[tuple[str, str]] = []
        for _ in range(max_merges):
            counts: Counter[tuple[str, str]] = Counter()
            for chunk in chunks:
                if chunk and not all(char.isspace() for char in chunk):
                    counts.update(zip(chunk, chunk[1:]))
            candidates = [
                (frequency, pair)
                for pair, frequency in counts.items()
                if frequency >= 3 and len(pair[0] + pair[1]) <= 12
            ]
            if not candidates:
                break
            # Deterministic frequency-first, Unicode lexical tie-break.
            best_pair = max(candidates, key=lambda item: (item[0], item[1]))[1]
            merges.append(best_pair)
            base.add(best_pair[0] + best_pair[1])
            chunks = [_merge_pair(chunk, best_pair) for chunk in chunks]
        return cls(sorted(base), merges)

    def encode(self, text: str) -> list[int]:
        output: list[int] = []
        for chunk in _bpe_chunks(text):
            for pair in self.merges:
                chunk = _merge_pair(chunk, pair)
            output.extend(self.char_to_id.get(token, UNK) for token in chunk)
        return output

    def decode(self, tokens: list[int]) -> str:
        return super().decode(tokens)
