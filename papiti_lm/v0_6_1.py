"""Papiti v0.6.1: brain-conditioned neural speaker for Hybrid.

Only the upstream Python/MaleCNS brain can select message content. The
Transformer must turn a completed semantic packet into Korean/English text.
No runtime intent classifier, arithmetic operation or canned answer exists.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import json
from pathlib import Path
import random
import sys


def _train(args):
    from papiti_lm.cli import _collate, _evaluate, _torch
    from papiti_lm.corpus import DigitSubwordTokenizer
    from papiti_lm.model import ModelConfig, PapitiTransformer, save_checkpoint
    from papiti_lm.v0_6_1_data import create_corpus, greeting_fingerprints
    from torch.utils.data import DataLoader

    torch = _torch()
    if not 1 <= args.epochs <= 60 or not 1 <= args.batch_size <= 96:
        raise ValueError("invalid training budget")
    if not 0 < args.learning_rate <= 0.01:
        raise ValueError("invalid learning rate")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    corpus = create_corpus(
        args.seed, diverse_social=args.diverse_social,
        everyday_dialogue=args.everyday_dialogue, phrasing_cues=args.diverse_social,
        compositional_greetings=args.compositional_greetings,
    )
    fingerprints = greeting_fingerprints(corpus) if args.compositional_greetings else {}
    tokenizer = DigitSubwordTokenizer.from_examples(corpus.train, max_merges=256)
    config = ModelConfig(
        tokenizer.vocab_size, context=args.context, d_model=args.d_model,
        heads=args.heads, layers=args.layers, dropout=0.1,
    )
    model = PapitiTransformer(config)
    training = [tokenizer.sequence(row, config.context) for row in corpus.train]
    validation = [tokenizer.sequence(row, config.context) for row in corpus.valid]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    print(
        f"v0.6.1 model-only speech | random init | "
        f"parameters={sum(p.numel() for p in model.parameters()):,} "
        f"| train={len(corpus.train)} | validation={len(corpus.valid)} "
        f"| untouched test={len(corpus.test)} | vocab={tokenizer.vocab_size}",
        flush=True,
    )
    best = float("inf")
    stale = 0
    for epoch in range(1, args.epochs + 1):
        loader = DataLoader(training, batch_size=args.batch_size, shuffle=True,
                            collate_fn=_collate,
                            generator=torch.Generator().manual_seed(args.seed + epoch))
        model.train()
        for inputs, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(inputs, targets)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        score = _evaluate(
            model, validation, args.batch_size,
            sum(len(row.assistant) + 1 for row in corpus.valid),
        )
        print(json.dumps({"epoch": epoch, "validation": score}, ensure_ascii=False), flush=True)
        if score["char_nll"] < best:
            best = score["char_nll"]
            stale = 0
            save_checkpoint(args.checkpoint, model, tokenizer, {
                "version": "v0.6.1",
                "role": "brain_conditioned_language_realizer_only",
                "initialization": "random_from_scratch",
                "diverse_social_targets": args.diverse_social,
                "phrasing_cues": args.diverse_social,
                "compositional_greetings": args.compositional_greetings,
                "greeting_fingerprints": fingerprints,
                "everyday_dialogue": args.everyday_dialogue,
                "seed": args.seed, "epoch": epoch,
                "val_char_nll": score["char_nll"],
                "train_count": len(corpus.train),
                "valid_count": len(corpus.valid),
            })
        else:
            stale += 1
        if stale >= args.patience:
            print(f"early_stop at epoch={epoch}", flush=True)
            break
    print(f"Selected checkpoint {args.checkpoint}; test split untouched", flush=True)


def _generate_greeting(model, tokenizer, packet: str) -> str:
    import torch
    from papiti_lm.corpus import EOS, RESERVED
    from papiti_lm.v0_6_1_data import greeting_sequence

    try:
        source, start = greeting_sequence(tokenizer, packet)
    except KeyError as exc:
        raise ValueError("source greeting contains an out-of-vocabulary character") from exc
    with torch.inference_mode():
        device = next(model.parameters()).device
        output = model(torch.tensor([source], device=device))[0, start:].argmax(dim=-1).tolist()
    if output[-1] != EOS or any(token < RESERVED for token in output[:-1]):
        raise ValueError("model did not generate a complete greeting")
    return tokenizer.decode(output[:-1])


def _eval(args):
    from hybrid_speech import reply_key
    from papiti_lm.cli import _evaluate, _torch
    from papiti_lm.model import generate, load_checkpoint
    from papiti_lm.v0_6_1_data import create_corpus, greeting_sequence

    _torch()
    model, tokenizer, metadata = load_checkpoint(args.checkpoint)
    if metadata.get("version") != "v0.6.1":
        raise ValueError("v0.6.1 checkpoint required")
    corpus = create_corpus(
        int(metadata["seed"]),
        diverse_social=metadata.get("diverse_social_targets") is True,
        mirror_greetings=metadata.get("mirror_greetings") is True,
        everyday_dialogue=metadata.get("everyday_dialogue") is True,
        phrasing_cues=metadata.get("phrasing_cues") is True,
        compositional_greetings=metadata.get("compositional_greetings") is True,
    )
    test_sequences = [
        greeting_sequence(tokenizer, row.user, row.assistant) if row.group == "mirror-greeting"
        else tokenizer.sequence(row, model.config.context)
        for row in corpus.test
    ]
    metric = _evaluate(
        model, test_sequences,
        args.batch_size, sum(len(row.assistant) + 1 for row in corpus.test),
    )
    training_replies = set(metadata.get("greeting_fingerprints", {}).get("replies", ()))
    results = []
    for row in corpus.test:
        if row.group == "mirror-greeting":
            try:
                answer = _generate_greeting(model, tokenizer, row.user)
            except ValueError:
                answer = ""
        else:
            answer = generate(model, tokenizer, row.user, max_new_tokens=64,
                              temperature=0, top_k=12)
        result = {
            "purpose": row.group,
            "expected": row.assistant,
            "generated": answer,
            "exact_match": answer.strip() == row.assistant,
        }
        if row.group == "compositional-greeting":
            result["matches_training_reply"] = reply_key(answer) in training_replies
        results.append(result)
    report = {
        "version": "v0.6.1", "test_metrics": metric,
        "unseen_exact_match_count": sum(row["exact_match"] for row in results),
        "test_count": len(results),
        "test_generations": results,
        "release_ready": False,  # Human semantic review is required.
        "limitations": [
            "A tiny, from-scratch speaker may not preserve arbitrary upstream meanings.",
            "Exact match on unseen values is not evidence of human-level language competence.",
            "Failed or ungrounded generations must not silently replace verified native answers.",
        ],
    }
    with open(args.report, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def _fresh_wording(
    plan, decode, avoid_reply_keys: tuple[str, ...], *,
    delivery_styles: tuple[str, ...] = (),
) -> str:
    from hybrid_speech import check_candidate

    # Keep ordinary requests cheap; repeated turns get a few more chances.
    if delivery_styles:
        styles = random.sample(delivery_styles, len(delivery_styles))
        guided = styles if avoid_reply_keys else styles[:2]
        # On a repeat, try each learned delivery directly before sampling.
        attempts = [(style, 0.0 if avoid_reply_keys else 0.32) for style in guided]
        attempts += [(None, 0.65), (None, 0.0)]
    else:
        temperatures = (0.5, 0.6, 0.65, 0.72, 0.72, 0.0) if avoid_reply_keys else (0.5, 0.62, 0.0)
        attempts = [(None, temperature) for temperature in temperatures]
    for style, temperature in attempts:
        answer = decode(style=style, temperature=temperature, top_k=5)
        if check_candidate(plan, answer, avoid_reply_keys=avoid_reply_keys) is None:
            return answer
    return ""


def _novel_greeting(plan, packet, decode, fingerprints, avoid_reply_keys) -> str:
    from hybrid_speech import check_candidate, reply_key
    from papiti_lm.v0_6_1_data import GREETING_CUES, greeting_packet, greeting_parts

    blocked = (*fingerprints["replies"], *avoid_reply_keys)
    openings, followups = set(fingerprints["openings"]), set(fingerprints["followups"])
    for opening, followup in random.sample(GREETING_CUES, len(GREETING_CUES)):
        answer = decode(greeting_packet(packet, opening, followup))
        parts = greeting_parts(answer)
        if (parts and reply_key(parts[0]) in openings and reply_key(parts[1]) in followups
                and check_candidate(plan, answer, avoid_reply_keys=blocked) is None):
            return answer
    return ""


@lru_cache(maxsize=4)
def _load_speaker(path: Path, _revision: tuple[int, int, int]):
    from papiti_lm.model import load_checkpoint

    return load_checkpoint(path)


def _speaker_model(checkpoint: str):
    path = Path(checkpoint).resolve()
    stat = path.stat()
    # Picking up a replaced checkpoint must not require a server restart.
    return _load_speaker(path, (stat.st_ino, stat.st_mtime_ns, stat.st_size))


def _infer_answer(plan, checkpoint, *, base_checkpoint=None, alternate_checkpoint=None,
                  max_new_tokens=96, avoid_reply_keys=()):
    """Only checkpoint-driven token prediction; does not load training examples."""
    from papiti_lm.model import generate
    from hybrid_speech import REPLY_HISTORY_LIMIT
    from papiti_lm.v0_6_1_data import (
        DELIVERY_STYLES, EVERYDAY_SPEECH_ACTS, delivery_packet, is_korean_greeting, packet_text,
    )
    from semantic_planner import INTENT_FACT_IDS

    packet = packet_text(plan)
    act = plan["speech_act"]
    model, tokenizer, metadata = _speaker_model(checkpoint)
    # The old companion learned to copy greetings and had no ordinary
    # greeting targets. Use the base speaker until it is retrained.
    if base_checkpoint and (
        (act == "return_greeting" and metadata.get("mirror_greetings") is True)
        or (act in INTENT_FACT_IDS and metadata.get("phrasing_cues") is not True)
    ):
        model, tokenizer, metadata = _speaker_model(base_checkpoint)
    if metadata.get("version") != "v0.6.1":
        raise ValueError("v0.6.1 semantic speaker checkpoint required")
    if act in EVERYDAY_SPEECH_ACTS and metadata.get("everyday_dialogue") is not True:
        raise ValueError("update the social checkpoint for everyday dialogue")
    if act == "return_greeting" and metadata.get("mirror_greetings") is True:
        raise ValueError("retrain the companion checkpoint for generative greetings")
    if metadata.get("compositional_greetings") is True and is_korean_greeting(plan):
        fingerprints = metadata.get("greeting_fingerprints")
        if not fingerprints:
            raise ValueError("compositional greeting checkpoint lacks training fingerprints")

        def decode_greeting(text):
            return generate(model, tokenizer, text, max_new_tokens=max_new_tokens, temperature=0)

        return _novel_greeting(
            plan, packet, decode_greeting, fingerprints,
            avoid_reply_keys[:REPLY_HISTORY_LIMIT],
        )

    def realize(speaker):
        speaker_model, speaker_tokenizer, speaker_metadata = speaker

        def decode(*, style=None, **options):
            text = delivery_packet(packet, style) if style else packet
            return generate(
                speaker_model, speaker_tokenizer, text, max_new_tokens=max_new_tokens, **options,
            )

        return _fresh_wording(
            plan, decode, tuple(avoid_reply_keys[:REPLY_HISTORY_LIMIT]),
            delivery_styles=DELIVERY_STYLES if speaker_metadata.get("phrasing_cues") is True else (),
        )

    if act == "state_exact_math_result":
        return generate(model, tokenizer, packet, max_new_tokens=max_new_tokens, temperature=0)

    speakers = [(model, tokenizer, metadata)]
    if avoid_reply_keys and alternate_checkpoint:
        try:
            alternate = _speaker_model(alternate_checkpoint)
        except (OSError, RuntimeError, ValueError, KeyError):
            # A missing or unusable optional companion cannot break the primary.
            alternate = None
        if alternate and alternate[2].get("version") == "v0.6.1" and alternate[2].get("phrasing_cues") is True:
            if metadata.get("phrasing_cues") is True:
                speakers.append(alternate)
            else:
                speakers.insert(0, alternate)
    for speaker in speakers:
        answer = realize(speaker)
        if answer:
            return answer
    return ""


def _configure_inference():
    from papiti_lm.cli import _torch

    # Small autoregressive steps get little benefit from extra threads, but
    # those threads compete with the router and web server on shared CPUs.
    _torch().set_num_threads(1)


def _infer(args):
    _configure_inference()
    answer = _infer_answer(
        json.loads(sys.stdin.read()), args.checkpoint,
        base_checkpoint=args.base_checkpoint, max_new_tokens=args.max_new_tokens,
        avoid_reply_keys=args.avoid_reply_key,
    )
    if answer:
        print(answer, flush=True)


def _infer_worker(_args):
    _configure_inference()
    for line in sys.stdin:
        try:
            result = {"answer": _infer_answer(**json.loads(line))}
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            result = {"error": type(exc).__name__}
        print(json.dumps(result, ensure_ascii=False), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Papiti v0.6.1 trained brain-to-speech Transformer")
    actions = parser.add_subparsers(dest="command", required=True)
    train = actions.add_parser("train", help="train from scratch on semantic packets")
    evaluation = actions.add_parser("eval", help="score packets withheld from training")
    infer = actions.add_parser("infer", help="read complete brain plan JSON on stdin")
    worker = actions.add_parser("infer-worker", help="serve brain plans over local stdio")
    for cmd in (train, evaluation, infer):
        cmd.add_argument("--checkpoint", default="artifacts/papiti_language_v0_6_1.pt")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--diverse-social", action="store_true")
    train.add_argument("--everyday-dialogue", action="store_true")
    train.add_argument("--compositional-greetings", action="store_true")
    train.add_argument("--epochs", type=int, default=12)
    train.add_argument("--patience", type=int, default=5)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=0.001)
    train.add_argument("--context", type=int, default=768)
    train.add_argument("--d-model", type=int, default=96)
    train.add_argument("--heads", type=int, default=4)
    train.add_argument("--layers", type=int, default=2)
    train.set_defaults(func=_train)
    evaluation.add_argument("--report", default="reports/papiti_v0_6_1.json")
    evaluation.add_argument("--batch-size", type=int, default=16)
    evaluation.set_defaults(func=_eval)
    infer.add_argument("--max-new-tokens", type=int, default=96)
    infer.add_argument("--base-checkpoint")
    infer.add_argument("--avoid-reply-key", action="append", default=[])
    infer.set_defaults(func=_infer)
    worker.set_defaults(func=_infer_worker)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
