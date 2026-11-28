"""Train-only data/token audit and matched generation diagnostic for v0.7.1.

No test set access, vocabulary filtering, or runtime changes.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter
from pathlib import Path

from papiti_lm.corpus import UNK
from scripts.train_papiti_foundation import PROMPTS, key, load_split

PROBES = {
    "ko": ("오늘은 ", "나는 학교에서 ", "친구와 함께 ", "내일 아침에는 ", "비가 내리면 "),
    "en": ("I think ", "Yesterday, ", "My friend ", "When it rains, ", "The little cat "),
}
# Additional diagnostic prefixes, not a sealed final exam.
NEW_PROBES = {
    "ko": ("도서관에서 ", "버스를 타고 ", "날씨가 좋으면 ", "책을 읽다가 ", "우리 가족은 "),
    "en": ("At the library, ", "Before breakfast, ", "The window ", "We decided to ", "After school, "),
}


def load_reference(root, path):
    import torch
    from papiti_lm.model import load_checkpoint
    torch.set_num_threads(min(4, torch.get_num_threads()))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    rows = load_split(root, manifest, "train")
    model, tokenizer, meta = load_checkpoint(path)
    sig = meta.get("signature", {})
    if sig.get("train") != manifest["splits"]["train"]["sha256"]:
        raise ValueError("checkpoint does not match the selected training corpus")
    if sig.get("valid") != manifest["splits"]["valid"]["sha256"]:
        raise ValueError("checkpoint does not match the selected validation corpus")
    return rows, model, tokenizer, meta


def audit(args):
    rows, _, tok, meta = load_reference(args.data_dir, args.reference)
    report = {"checkpoint": str(args.reference), "epoch": meta["epoch"],
              "train_only": True, "languages": {}, "human_review": []}
    rng = random.Random(71042)
    for lang in PROMPTS:
        subset = [r for r in rows if r["lang"] == lang]
        counts = Counter()
        lengths, chars, tokens = [], 0, 0
        for row in subset:
            text = row["text"]
            ids = tok.encode(text)
            chars += len(text)
            tokens += len(ids)
            lengths.append(len(ids))
            counts.update(ids)
        report["languages"][lang] = {
            "sentences": len(subset), "chars": chars, "tokens": tokens,
            "chars_per_token": chars / tokens,
            "median_tokens_per_sentence": statistics.median(lengths),
            "unknown_tokens": counts[UNK],
            "most_frequent_tokens": [
                [tok.decode([i]), n] for i, n in counts.most_common(20)],
        }
        for row in rng.sample(subset, min(args.sample, len(subset))):
            report["human_review"].append({
                "lang": lang, "id": row["id"], "text": row["text"],
                "source": row.get("source"), "author": row.get("author"),
                "license": row.get("license"),
                "tokens": [tok.decode([i]) for i in tok.encode(row["text"])],
                "grammar_ok": None, "natural": None, "appropriate": None,
            })
    report["diagnostic_fragments"] = [
        {"text": s, "tokens": [tok.decode([i]) for i in tok.encode(s)]}
        for s in ("brumper", "bruntil", "blude", "bludent", "Hichichance",
                  "그 사람은", "서처음", "I want to play tonight.")
    ]
    # Re-encoding broken words illustrates segmentation; it does not recover
    # their original generation token paths.
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"languages": report["languages"],
                      "sample_count": len(report["human_review"]),
                      "diagnostic_fragments": report["diagnostic_fragments"],
                      "report": str(args.report),
                      "human_review_required": True}, ensure_ascii=False))


def compare(args):
    from papiti_lm.model import generate
    results = []
    reference_vocab = None
    reference_signature = None
    for path in (args.reference, args.candidate):
        rows, model, tok, meta = load_reference(args.data_dir, path)
        vocab = (tok.characters, tok.merges)
        signature = {k: v for k, v in meta["signature"].items() if k != "width"}
        if reference_vocab is None:
            reference_vocab, reference_signature = vocab, signature
        elif vocab != reference_vocab or signature != reference_signature:
            raise ValueError("comparison requires same tokenizer/data/settings except width")
        known = {key(r["text"]) for r in rows}
        records = []
        for lang in PROMPTS:
            cases = [("old_free", "", t, s) for t, s in
                     [(0, 42), (.8, 42), (.8, 123), (.8, 2026)]]
            cases += [("old_completion", p, 0, 42) for p in PROBES[lang]]
            cases += [("new_completion", p, 0, 42) for p in NEW_PROBES[lang]]
            cases += [("new_free", "", .8, s) for s in (71, 710, 7100)]
            for group, prefix, temp, seed in cases:
                answer = generate(model, tok, PROMPTS[lang],
                                  response_prefix=prefix, max_new_tokens=80,
                                  temperature=temp, top_k=20, seed=seed,
                                  repetition_penalty=1.)
                record = {"group": group, "lang": lang, "prefix": prefix,
                          "temperature": temp, "seed": seed, "output": answer,
                          "copied_training_sentence": key(answer) in known,
                          "ends_in_punctuation": answer.rstrip().endswith((".", "?", "!", "。", "？", "！")),
                          "human_fluent": None}
                records.append(record)
                print(json.dumps({"width": model.config.d_model, **record},
                                 ensure_ascii=False), flush=True)
        results.append({"checkpoint": str(path), "epoch": meta["epoch"],
                        "width": model.config.d_model,
                        "parameters": sum(p.numel() for p in model.parameters()),
                        "macro_char_nll": meta["macro_char_nll"], "records": records})
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"results": results, "test_untouched": True,
                                      "human_review_required": True},
                                     ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Comparison saved:", args.report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["audit", "compare"])
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/papiti_v070_data"))
    parser.add_argument("--reference", type=Path, default=Path("artifacts/papiti_v070_foundation/best.pt"))
    parser.add_argument("--candidate", type=Path, default=Path("artifacts/papiti_v071_capacity/best.pt"))
    parser.add_argument("--report", type=Path)
    parser.add_argument("--sample", type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.sample <= 500:
        parser.error("sample must be 1..500 per language")
    if args.report is None:
        args.report = Path("artifacts/papiti_v071_" + args.mode + ".json")
    (audit if args.mode == "audit" else compare)(args)


if __name__ == "__main__":
    main()
