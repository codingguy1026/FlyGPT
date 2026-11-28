"""Papiti v0.7 phase 1: bilingual sentence-level language pretraining.

Download only Tatoeba's explicit CC0 export, or use an explicitly supplied
local copy of that archive. No user chats, pretrained weights or answer
lookup. This is a research checkpoint, not a production assistant.
"""
from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import random
import re
import shutil
import tarfile
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path

from papiti_lm.corpus import (
    Example, DigitSubwordTokenizer, UNK, _merge_pair,
)

URL = "https://downloads.tatoeba.org/exports/sentences_CC0.tar.bz2"
LANGS = {"kor": "ko", "eng": "en"}
PROMPTS = {"ko": "Language: ko\nWrite a sentence.",
           "en": "Language: en\nWrite a sentence."}


def key(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def acceptable(lang, text):
    # Structural/obvious contact-information checks on the DATA, not a runtime
    # word blacklist. These do not establish grammatical or content quality.
    if not 12 <= len(text) <= 160 or any(unicodedata.category(c)[0] == "C" for c in text):
        return False
    if re.search(r"https?://|www\.|@|\d{3}[- .]\d{3,4}[- .]\d{4}", text, re.I):
        return False
    if not text.endswith((".", "?", "!", "。", "！", "？")):
        return False
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 5:
        return False
    if lang == "ko":
        return sum("\uac00" <= c <= "\ud7a3" for c in letters) / len(letters) >= .85
    return sum(c.isascii() for c in letters) / len(letters) >= .95


def legacy_exclusions():
    from papiti_lm.inquiry_curriculum import add_inquiry_training
    from papiti_lm.v0_6_1_prose import create_prose_corpus
    corpus = add_inquiry_training(create_prose_corpus(seed=42, repeats=2))
    # Exclude exact old utterances and constituent lines from every split,
    # including training, to reduce contamination of legacy transfer exams.
    # No excluded texts are added to training or the tokenizer.
    return {key(part) for row in (*corpus.train, *corpus.valid, *corpus.test)
            for part in (row.assistant, *row.assistant.splitlines())}


def read_cc0(archive, excluded):
    seen = set(excluded)
    by_lang = {lang: [] for lang in LANGS.values()}
    with tarfile.open(archive, "r:bz2") as tar:
        members = [m for m in tar.getmembers() if m.isfile()
                   and m.name.lower().endswith((".csv", ".tsv", ".txt"))]
        if len(members) != 1:
            raise ValueError("unexpected CC0 archive layout; inspect source before use")
        stream = tar.extractfile(members[0])
        if stream is None:
            raise ValueError("missing sentence export")
        for raw in stream:
            fields = raw.decode("utf-8-sig").rstrip("\r\n").split("\t", 2)
            if len(fields) != 3 or not fields[0].isdigit() or fields[1] not in LANGS:
                continue
            sid, code, text = fields
            lang = LANGS[code]
            text = unicodedata.normalize("NFC", text.strip())
            ident = key(text)
            if ident in seen or not acceptable(lang, text):
                continue
            seen.add(ident)
            by_lang[lang].append({"id": sid, "lang": lang, "text": text,
                                  "license": "CC0-1.0",
                                  "source": "https://tatoeba.org/en/sentences/show/" + sid})
    return by_lang


def supplement_attributed(root, pool, excluded, target):
    """Opt-in CC BY fallback using official per-language detailed exports."""
    seen = set(excluded)
    seen.update(key(row["text"]) for rows in pool.values() for row in rows)
    sources = []
    for code, lang in LANGS.items():
        if len(pool[lang]) >= target:
            continue
        filename = code + "_sentences_detailed.tsv.bz2"
        url = "https://downloads.tatoeba.org/exports/per_language/" + code + "/" + filename
        path = root / filename
        if not path.exists():
            tmp = path.with_suffix(".download")
            try:
                with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as out:
                    shutil.copyfileobj(response, out)
                tmp.replace(path)
            finally:
                tmp.unlink(missing_ok=True)
        sources.append({"url": url, "sha256": digest(path),
                        "license": "CC-BY-2.0-FR (CC0 entries may also be present)"})
        # Reservoir sampling over eligible rows; avoids loading the entire
        # English detailed export into memory. Reserve exactly the shortfall.
        needed = target - len(pool[lang])
        rng = random.Random(42 + (0 if lang == "ko" else 1))
        candidates = []
        eligible = 0
        with bz2.open(path, "rt", encoding="utf-8-sig") as stream:
            for line in stream:
                fields = line.rstrip("\r\n").split("\t")
                if len(fields) != 6 or not fields[0].isdigit() or fields[1] != code:
                    continue
                sid, _, raw, author, _, _ = fields
                text = unicodedata.normalize("NFC", raw.strip())
                ident = key(text)
                if not author or author == r"\N" or ident in seen or not acceptable(lang, text):
                    continue
                seen.add(ident)
                row = {"id": sid, "lang": lang, "text": text,
                       "license": "CC-BY-2.0-FR", "author": author,
                       "source": "https://tatoeba.org/en/sentences/show/" + sid}
                eligible += 1
                if len(candidates) < needed:
                    candidates.append(row)
                else:
                    slot = rng.randrange(eligible)
                    if slot < needed:
                        candidates[slot] = row
        pool[lang].extend(candidates)
        print(json.dumps({"stage": "attributed_supplement", "lang": lang,
                          "eligible": eligible, "added": len(candidates)}), flush=True)
    return sources


def prepare(args):
    root = args.data_dir
    root.mkdir(parents=True, exist_ok=True)
    archive = args.archive or root / "sentences_CC0.tar.bz2"
    if not archive.exists():
        if args.archive:
            raise FileNotFoundError(archive)
        tmp = archive.with_suffix(".download")
        try:
            with urllib.request.urlopen(URL, timeout=120) as response, tmp.open("wb") as out:
                shutil.copyfileobj(response, out)
            tmp.replace(archive)
        finally:
            tmp.unlink(missing_ok=True)
    print(json.dumps({"stage": "prepare", "archive_sha256": digest(archive)}), flush=True)
    excluded = legacy_exclusions()
    pool = read_cc0(archive, excluded)
    print(json.dumps({"stage": "cc0_counts", "usable": {k: len(v) for k, v in pool.items()}}), flush=True)
    sources = []
    if args.allow_cc_by:
        sources = supplement_attributed(root, pool, excluded, args.per_language)
    count = min(args.per_language, *(len(v) for v in pool.values()))
    if count < 200:
        raise ValueError("fewer than 200 usable sentences per language: " + str({k: len(v) for k, v in pool.items()}) + "; optionally use prepare --allow-cc-by for attributed Tatoeba exports")
    rng = random.Random(args.seed)
    splits = {"train": [], "valid": [], "test": []}
    for lang, rows in sorted(pool.items()):
        # Deterministic even if the source archive's line order changes.
        rows.sort(key=lambda row: int(row["id"]))
        rng.shuffle(rows)
        rows = rows[:count]
        n = max(1, count // 20)
        splits["test"].extend(rows[:n])
        splits["valid"].extend(rows[n:2*n])
        splits["train"].extend(rows[2*n:])
    for name, rows in splits.items():
        rng.shuffle(rows)
        path = root / (name + ".jsonl")
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as out:
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(path)
    manifest = {
        "source": URL, "license": "CC0-1.0 + CC-BY-2.0-FR" if sources else "CC0-1.0", "seed": args.seed,
        "supplement_sources": sources,
        "archive_sha256": digest(archive),
        "available_after_structural_filter": {k: len(v) for k, v in pool.items()},
        "requested_per_language": args.per_language, "selected_per_language": count,
        "splits": {name: {"count": len(rows), "sha256": digest(root / (name + ".jsonl"))}
                   for name, rows in splits.items()},
        "limitations": "Exact normalized deduplication only; not human-reviewed; no guarantee of fluency, safety or semantic disjointness.",
    }
    (root / "ATTRIBUTION.txt").write_text(
        "Sentences contributed by Tatoeba contributors: https://tatoeba.org\n"
        "CC0: https://creativecommons.org/publicdomain/zero/1.0/\n"
        "CC BY 2.0 FR: https://creativecommons.org/licenses/by/2.0/fr/\n"
        "Each JSONL record contains its source URL and license; attributed rows include author.\n"
        "Data selected, NFC-normalized, whitespace-trimmed and deduplicated.\n"
        "Retain this notice and record attribution when redistributing data.\n",
        encoding="utf-8",
    )
    tmp = root / "manifest.tmp"
    tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    tmp.replace(root / "manifest.json")
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


def load_split(root, manifest, name):
    path = root / (name + ".jsonl")
    if digest(path) != manifest["splits"][name]["sha256"]:
        raise ValueError("data changed after preparation: " + name)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != manifest["splits"][name]["count"]:
        raise ValueError("split size mismatch")
    return rows


def examples(rows):
    return [Example("foundation-" + r["lang"], PROMPTS[r["lang"]], r["text"])
            for r in rows]


def train_tokenizer(rows, max_merges):
    # Weighted unique chunks: equivalent frequency-count idea to existing BPE,
    # but avoids rescanning millions of repeated words from scratch.
    chunks = Counter()
    base = set()
    for row in rows:
        for text in (row.user, row.assistant):
            base.update(text)
            chunks.update(tuple(c) for c in DigitSubwordTokenizer._chunks(text))
    merges = []
    for _ in range(max_merges):
        counts = Counter()
        for chunk, freq in chunks.items():
            if len(chunk) > 1 and not all(c.isspace() for c in chunk):
                for pair in zip(chunk, chunk[1:]):
                    if len(pair[0] + pair[1]) <= 12:
                        counts[pair] += freq
        eligible = [(freq, pair) for pair, freq in counts.items() if freq >= 3]
        if not eligible:
            break
        _, best = max(eligible)
        merges.append(best)
        base.add(best[0] + best[1])
        updated = Counter()
        for chunk, freq in chunks.items():
            updated[tuple(_merge_pair(list(chunk), best))] += freq
        chunks = updated
    return DigitSubwordTokenizer(sorted(base), merges)


def atomic_state(torch, path, state):
    tmp = path.with_suffix(".tmp")
    try:
        torch.save(state, tmp)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def train(args):
    import torch
    from torch.utils.data import DataLoader
    from papiti_lm.cli import _collate, _evaluate
    from papiti_lm.model import ModelConfig, PapitiTransformer, save_checkpoint

    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(args.seed)
    root = args.data_dir
    manifest = json.loads((root / "manifest.json").read_text())
    train_rows = load_split(root, manifest, "train")
    val_rows = load_split(root, manifest, "valid")
    # The test text is never loaded here. Do not use it for checkpoint selection.
    tr_keys = {key(r["text"]) for r in train_rows}
    if len(tr_keys) != len(train_rows) or tr_keys & {key(r["text"]) for r in val_rows}:
        raise ValueError("duplicate or train/validation overlap")
    tr, va = examples(train_rows), examples(val_rows)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    last, best_path = out / "latest.pt", out / "best.pt"
    signature = {"train": manifest["splits"]["train"]["sha256"],
                 "valid": manifest["splits"]["valid"]["sha256"],
                 "seed": args.seed, "merges": args.merges, "batch": args.batch_size,
                 "lr": args.lr, "context": 256, "width": args.width, "layers": 3}
    state = None
    if last.exists():
        if not args.resume:
            raise ValueError("latest.pt exists; use --resume or choose another output dir")
        state = torch.load(last, map_location="cpu", weights_only=True)
        if state["signature"] != signature:
            raise ValueError("resume data/settings mismatch")
        tok = DigitSubwordTokenizer(state["tokens"], state["merges"])
    else:
        if args.resume or best_path.exists():
            raise ValueError("no resumable latest checkpoint; preserve best.pt and use a new output dir")
        print(json.dumps({"stage": "learn_train_only_bpe", "train": len(tr)}), flush=True)
        tok = train_tokenizer(tr, args.merges)
    model = PapitiTransformer(ModelConfig(tok.vocab_size, context=256,
                                         d_model=args.width, heads=4, layers=3))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    start, best, best_epoch = 1, float("inf"), 0
    if state is not None:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng"])
        start, best, best_epoch = state["epoch"] + 1, state["best"], state["best_epoch"]
    train_seq = [tok.sequence(x, model.config.context) for x in tr]
    valid_seq = {lang: [tok.sequence(x, model.config.context) for x in va
                        if x.group == "foundation-" + lang] for lang in PROMPTS}
    valid_chars = {lang: sum(len(x.assistant)+1 for x in va
                            if x.group == "foundation-" + lang) for lang in PROMPTS}
    unknown = {lang: sum(y == UNK for _, ys in seq for y in ys)
               for lang, seq in valid_seq.items()}
    experiment = "papiti_v070_foundation" if args.width == 128 else "papiti_v071_capacity"
    print(json.dumps({"experiment": experiment, "width": args.width, "start_epoch": start,
                      "parameters": sum(p.numel() for p in model.parameters()),
                      "vocab": tok.vocab_size, "train": len(tr), "valid": len(va),
                      "unknown_validation_targets": unknown,
                      "frozen_test_untouched": True}), flush=True)
    for epoch in range(start, args.epochs + 1):
        loader = DataLoader(train_seq, batch_size=args.batch_size, shuffle=True,
                            collate_fn=_collate,
                            generator=torch.Generator().manual_seed(args.seed + epoch))
        model.train()
        total, count = 0., 0
        for step, (inputs, targets) in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(inputs, targets)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            n = int(targets.ne(-100).sum())
            total += float(loss.detach()) * n
            count += n
            if step % 50 == 0:
                print(json.dumps({"epoch": epoch, "step": step, "steps": len(loader),
                                  "train_nll_so_far": round(total/count, 5)}), flush=True)
        metrics = {lang: _evaluate(model, seq, args.batch_size, valid_chars[lang])
                   for lang, seq in valid_seq.items()}
        score = sum(m["char_nll"] for m in metrics.values()) / len(metrics)
        print(json.dumps({"epoch": epoch, "stage": "language_foundation",
                          "train_nll": round(total/count, 5), "validation": metrics,
                          "macro_char_nll": round(score, 5)}), flush=True)
        if score < best:
            best, best_epoch = score, epoch
            save_checkpoint(best_path, model, tok, {
                "version": "v0.7-foundation" if args.width == 128 else "v0.7.1-capacity",
                "role": "sentence_pretraining_only",
                "epoch": epoch, "macro_char_nll": score, "signature": signature,
                "source": URL, "license": manifest["license"],
                "supplement_sources": manifest.get("supplement_sources", []),
                "initialization": "random_from_scratch",
            })
        atomic_state(torch, last, {
            "epoch": epoch, "best": best, "best_epoch": best_epoch,
            "signature": signature, "tokens": tok.characters, "merges": tok.merges,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "rng": torch.get_rng_state(),
        })
    print(json.dumps({"selected_checkpoint": str(best_path), "best_epoch": best_epoch,
                      "best_macro_char_nll": best, "resume_checkpoint": str(last),
                      "frozen_test_untouched": True}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--data-dir", type=Path, default=Path("artifacts/papiti_v070_data"))
    prep.add_argument("--archive", type=Path)
    prep.add_argument("--allow-cc-by", action="store_true",
                      help="Opt in to attributed CC BY 2.0 FR Tatoeba supplements")
    prep.add_argument("--per-language", type=int, default=5000)
    prep.add_argument("--seed", type=int, default=42)
    tr = sub.add_parser("train")
    tr.add_argument("--data-dir", type=Path, default=Path("artifacts/papiti_v070_data"))
    tr.add_argument("--output-dir", type=Path, default=Path("artifacts/papiti_v070_foundation"))
    tr.add_argument("--epochs", type=int, default=8)
    tr.add_argument("--batch-size", type=int, default=24)
    tr.add_argument("--merges", type=int, default=512)
    tr.add_argument("--lr", type=float, default=.0007)
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--width", type=int, choices=[128, 192], default=128)
    tr.add_argument("--resume", action="store_true")
    a = p.parse_args()
    if a.command == "prepare":
        if not 200 <= a.per_language <= 25000:
            p.error("per-language must be 200..25000")
        prepare(a)
    else:
        if not 1 <= a.epochs <= 100 or not 1 <= a.batch_size <= 128 or not 0 <= a.merges <= 1024 or not 0 < a.lr <= .01:
            p.error("invalid training settings")
        if a.width != 128 and a.output_dir == Path("artifacts/papiti_v070_foundation"):
            p.error("use a separate --output-dir for width 192")
        train(a)


if __name__ == "__main__":
    main()
