"""Offline evaluation; rubric targets never enter autoregressive generation."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

from papiti_lm.corpus import Example, UNK


def surface_metrics(text: str) -> dict:
    words = text.split()
    triples = list(zip(words, words[1:], words[2:]))
    repeated = sum(count - 1 for count in Counter(triples).values())
    return {
        "empty": not bool(text.strip()),
        "malformed_unicode": "\ufffd" in text or any(
            0x1100 <= ord(character) <= 0x11ff for character in text),
        "repeated_trigram_fraction": repeated / max(1, len(triples)),
        "sentence_count": len(re.findall(r"[.!?]+(?:\s|$)", text)),
    }


def normalized_answer(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).casefold()


def conditional_nll(model, tokenizer, examples: list[Example]):
    import torch
    from torch.nn import functional
    from papiti_lm.cli import _collate

    inputs, targets = _collate([tokenizer.sequence(example, model.config.context)
                               for example in examples])
    logits = model(inputs)
    losses = functional.cross_entropy(logits.transpose(1, 2), targets,
                                     ignore_index=-100, reduction="none")
    return losses.sum(1) / targets.ne(-100).sum(1)


def contrast_pairs(curriculum, split: str) -> list[tuple[Example, Example]]:
    key = "contrasts" if split == "train" else "test_contrasts"
    rows = curriculum.select(split, "language")
    by_family = {record.family: record.example for record in rows}
    pairs = []
    for left, right in curriculum.manifest[key]:
        for positive, negative in ((left, right), (right, left)):
            example = by_family[positive]
            pairs.append((example, Example(example.group, example.user,
                                           by_family[negative].assistant)))
    return pairs


def evaluate_checkpoint(checkpoint: str, curriculum, batch_size: int = 24) -> dict:
    import torch
    from papiti_lm.cli import _evaluate
    from papiti_lm.model import generate, load_checkpoint

    model, tokenizer, metadata = load_checkpoint(checkpoint)
    if metadata.get("version") != "v0.6" or metadata.get("source_sha256") != curriculum.digest:
        raise ValueError("v0.6 checkpoint/curriculum fingerprint mismatch")
    report = {
        "schema_version": 1, "checkpoint": checkpoint, "metadata": metadata,
        "decoding": {"temperature": 0, "max_new_tokens": 128, "repetition_penalty": 1.0},
        "split": "test (never used for tokenizer, optimization or checkpoint selection)",
        "metrics": {}, "generations": [], "human_review_complete": False,
        "release_ready": False,
        "limitations": [
            "One seed and a small original curriculum cannot establish general reasoning.",
            "Reference likelihood and keyword checks do not prove grammatical or contextual correctness.",
            "Knowledge tests hold out entire formula families, not just paraphrases.",
            "No human ratings have been filled automatically; inspect every generated response.",
            "Pipeline success is not a model-quality promotion gate.",
        ],
    }
    for domain in ("language", "arithmetic", "knowledge"):
        rows = curriculum.select("test", domain)
        examples = [record.example for record in rows]
        metrics = _evaluate(model, [tokenizer.sequence(example, model.config.context)
                                   for example in examples], batch_size,
                            sum(len(example.assistant) + 1 for example in examples))
        encoded = [tokenizer.encode(example.assistant) for example in examples]
        metrics["target_unknown_token_fraction"] = sum(tokens.count(UNK) for tokens in encoded) / max(
            1, sum(map(len, encoded)))
        metrics["test_examples"] = len(rows)
        metrics["exact_matches"] = 0
        metrics["generated_examples"] = 0
        metrics["keyword_rubric_passes"] = 0
        metrics["empty_outputs"] = 0
        metrics["malformed_outputs"] = 0
        metrics["outputs_with_repeated_trigrams"] = 0
        for record in rows:
            if domain == "language" and record.source != "original":
                continue
            example = record.example
            answer = generate(model, tokenizer, example.user, temperature=0,
                              max_new_tokens=128)
            exact = normalized_answer(answer) == normalized_answer(example.assistant)
            surface = surface_metrics(answer)
            rubric = (bool(record.required_any)
                      and any(keyword in answer for keyword in record.required_any)
                      and not any(keyword in answer for keyword in record.forbidden))
            metrics["generated_examples"] += 1
            metrics["exact_matches"] += int(exact)
            metrics["keyword_rubric_passes"] += int(rubric)
            metrics["empty_outputs"] += int(surface["empty"])
            metrics["malformed_outputs"] += int(surface["malformed_unicode"])
            metrics["outputs_with_repeated_trigrams"] += int(surface["repeated_trigram_fraction"] > 0)
            report["generations"].append({
                "domain": domain, "family": record.family, "prompt": example.user,
                "prediction": answer, "reference": example.assistant,
                "source": record.source, "exact_match": exact,
                "keyword_rubric_pass": rubric, "surface": surface,
                "human_scores_0_to_2": {
                    "grammar_and_word_order": None, "context_and_intent": None,
                    "coherence_and_repetition": None, "factual_correctness": None,
                },
            })
        metrics["exact_match_fraction"] = metrics["exact_matches"] / max(1, metrics["generated_examples"])
        report["metrics"][domain] = metrics
    pairs = contrast_pairs(curriculum, "test")
    with torch.inference_mode():
        preferences = []
        for positive, negative in pairs:
            scores = conditional_nll(model, tokenizer, [positive, negative]).tolist()
            preferences.append({"prompt": positive.user, "appropriate_nll": scores[0],
                                "opposite_context_nll": scores[1], "correct": scores[0] < scores[1]})
    report["contrastive_context_preferences"] = preferences
    return report
