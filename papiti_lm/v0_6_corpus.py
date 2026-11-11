"""Offline, provenance-bearing curriculum with separate development and test sets."""
from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from papiti_lm.corpus import Example, load_examples, split_by_group
from papiti_lm.v0_5_corpus import arithmetic_examples, augment_language

DATA = "training/papiti_v0_6_curriculum.json"
LEGACY = "training/papiti_dialogues_v0_3.jsonl"


def normalize_prompt(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"[\W_]+", "", value)


@dataclass(frozen=True)
class Record:
    example: Example
    domain: str
    family: str
    split: str
    level: int = 0
    source: str = "original"
    required_any: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()


@dataclass
class Curriculum:
    records: list[Record]
    manifest: dict
    digest: str

    def select(self, split: str, domain: str | None = None) -> list[Record]:
        return [record for record in self.records if record.split == split
                and (domain is None or record.domain == domain)]


def audit(records: list[Record]) -> None:
    families: dict[str, str] = {}
    prompts: dict[str, tuple[str, str]] = {}
    for record in records:
        if record.split not in {"train", "valid", "test"}:
            raise ValueError("invalid split")
        if record.domain not in {"language", "arithmetic", "knowledge"}:
            raise ValueError("invalid domain")
        if not record.family or not record.example.user.strip() or not record.example.assistant.strip():
            raise ValueError("empty curriculum field")
        previous_split = families.setdefault(record.family, record.split)
        if previous_split != record.split:
            raise ValueError(f"family leakage: {record.family}")
        key = normalize_prompt(record.example.user)
        previous = prompts.setdefault(key, (record.split, record.example.assistant))
        if previous[0] != record.split:
            raise ValueError(f"normalized prompt leakage: {record.example.user}")
    for split in ("train", "valid", "test"):
        for domain in ("language", "arithmetic", "knowledge"):
            if not any(record.split == split and record.domain == domain for record in records):
                raise ValueError(f"empty {split}/{domain}")


def connected_groups(examples: list[Example]) -> list[Example]:
    parents = {example.group: example.group for example in examples}

    def root(group: str) -> str:
        while parents[group] != group:
            parents[group] = parents[parents[group]]
            group = parents[group]
        return group

    owners: dict[str, str] = {}
    for example in examples:
        key = normalize_prompt(example.user)
        owner = owners.setdefault(key, example.group)
        left, right = root(owner), root(example.group)
        parents[max(left, right)] = min(left, right)
    return [Example(root(example.group), example.user, example.assistant) for example in examples]


def prepare(data: str = DATA, legacy: str = LEGACY, seed: int = 42,
            math_train: int = 1200, math_holdout: int = 80) -> Curriculum:
    path, old_path = Path(data), Path(legacy)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported curriculum schema")
    records: list[Record] = []
    for item in manifest["language"] + manifest["knowledge"]:
        source = item["source"]
        if source not in manifest["sources"]:
            raise ValueError(f"missing provenance: {source}")
        if not 0 <= item["level"] <= 3:
            raise ValueError("invalid curriculum level")
        for prompt in item["prompts"]:
            records.append(Record(
                Example(item["family"], prompt, item["answer"]),
                item["domain"], item["family"], item["split"], item["level"], source,
                tuple(item.get("required_any", [])), tuple(item.get("forbidden", [])),
            ))
    curated_prompts = {normalize_prompt(record.example.user) for record in records}
    original = connected_groups(load_examples(old_path))
    development, test = split_by_group(original, seed, 0.18)
    train, valid = split_by_group(development, seed + 1, 0.22)
    for split, examples in (("train", train), ("valid", valid), ("test", test)):
        for example in examples:
            if normalize_prompt(example.user) not in curated_prompts:
                records.append(Record(example, "language", "legacy:" + example.group, split, source="legacy"))
    arithmetic = arithmetic_examples(seed, math_train, math_holdout * 2)
    for split, examples in (("train", arithmetic.train),
                            ("valid", arithmetic.valid[:math_holdout]),
                            ("test", arithmetic.valid[math_holdout:])):
        for example in examples:
            operands = sorted(int(number) for number in re.findall(r"\d+", example.user))
            family = "addition:" + ":".join(map(str, operands))
            records.append(Record(example, "arithmetic", family, split, source="synthetic"))
    audit(records)
    digest = hashlib.sha256(path.read_bytes() + b"\0" + old_path.read_bytes()
                            + Path(__file__).read_bytes()
                            + Path(__file__).with_name("v0_5_corpus.py").read_bytes()).hexdigest()
    return Curriculum(records, manifest, digest)


def training_pools(curriculum: Curriculum) -> dict[str, list[Example]]:
    pools = {domain: [record.example for record in curriculum.select("train", domain)]
             for domain in ("language", "arithmetic", "knowledge")}
    variants = augment_language(pools["language"])
    heldout = {normalize_prompt(record.example.user) for record in curriculum.records
               if record.split != "train"}
    pools["language"] += [example for example in variants
                          if normalize_prompt(example.user) not in heldout]
    return pools


def epoch_examples(curriculum: Curriculum, epoch: int, seed: int) -> list[Example]:
    if epoch < 1:
        raise ValueError("epoch must be positive")
    pools = training_pools(curriculum)
    allowed = 1 if epoch <= 2 else 2 if epoch <= 4 else 3
    pools["knowledge"] = [record.example for record in curriculum.select("train", "knowledge")
                          if record.level <= allowed]
    rng = random.Random(seed + epoch)
    count = len(pools["language"])
    output = list(pools["language"])
    for domain, fraction in (("arithmetic", 0.5), ("knowledge", 0.5)):
        pool = pools[domain]
        if not pool:
            raise ValueError(f"empty curriculum stage: {domain}")
        sample_count = max(1, round(count * fraction))
        repetitions, remainder = divmod(sample_count, len(pool))
        output.extend(pool * repetitions + rng.sample(pool, remainder))
    rng.shuffle(output)
    return output
