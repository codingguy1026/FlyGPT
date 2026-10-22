"""Experimental supervised rule-priority head over frozen MaleCNS propagation.

This is a research script, NOT deployed. Run from repo root:
python scripts/train_malecns_rule_head.py
Requires torch and the real configured checkpoint. Never presents synthetic
training labels or an untrained network as established biological reasoning.
"""
from __future__ import annotations
import json
import random
import statistics
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chain_reasoner import solve_chain


def make_examples(count: int, seed: int):
    """Independent train/test seeds; balanced label, target, and rule order."""
    rng = random.Random(seed)
    samples = []
    # The supervision target is which dependency must be resolved immediately
    # to prove the requested atom, NOT a language-route class.
    for i in range(count):
        conjunction_first = i % 2 == 0
        target = rng.choice("EGH")
        if conjunction_first:
            rules = [f"A와 B가 모두 참이면 {target}가 참이다",
                     "A이면 B이다", "A는 참이다"]
        else:
            rules = [f"A이면 {target}이다",
                     "A와 B가 모두 참이면 D가 참이다", "A는 참이다"]
        # Include irrelevant rules to ensure task needs dependency reasoning.
        rules.insert(rng.randint(0, len(rules)), "X이면 Y이다")
        # Always put facts before the question, and reorder only rules.
        facts = [x for x in rules if x.endswith("는 참이다")]
        dependencies = [x for x in rules if x not in facts]
        rng.shuffle(dependencies)
        question = ". ".join(dependencies + facts + [f"{target}는 참인가?"])
        solved = solve_chain(question)
        assert solved is not None and solved.evidence["entailed"]
        samples.append((question, int(conjunction_first)))
    rng.shuffle(samples)
    return samples


def embed(router, examples):
    import torch
    from training.train_router import vectorize
    with torch.no_grad():
        batch = torch.stack([vectorize(t, router.vocab_size, router.vectorizer_version)
                             for t, _ in examples])
        # Freeze actual pretrained MaleCNS encoder and graph propagation.
        return router.model._readout(router.model.propagate(batch)).detach()


def fit(train_x, train_y, seed=42):
    import torch
    torch.manual_seed(seed)
    head = torch.nn.Linear(train_x.shape[1], 2)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.007, weight_decay=.03)
    for epoch in range(75):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(head(train_x), train_y)
        loss.backward()
        optimizer.step()
    return head


def accuracy(head, x, labels):
    import torch
    with torch.no_grad():
        predicted = head(x).argmax(dim=1)
        return float((predicted == labels).float().mean()), predicted.tolist()


def main():
    import torch
    from scripts.benchmark_neural_rule_scheduler import load_router
    router = load_router()
    train = make_examples(160, 8123)
    heldout = make_examples(100, 92371)
    # Most prompts share a compact grammar; this is a template-family
    # generalization check, not evidence of open-domain reasoning.
    xtr = embed(router, train)
    xte = embed(router, heldout)
    ytr = torch.tensor([y for _, y in train], dtype=torch.long)
    yte = torch.tensor([y for _, y in heldout], dtype=torch.long)
    head = fit(xtr, ytr)
    score, labels = accuracy(head, xte, yte)
    # Randomly initialized head control with same embeddings.
    torch.manual_seed(987)
    random_head = torch.nn.Linear(xte.shape[1], 2)
    control, _ = accuracy(random_head, xte, yte)
    # Shuffle gold labels to test that training signal matters.
    perm = torch.randperm(len(ytr), generator=torch.Generator().manual_seed(22))
    shuffled_head = fit(xtr, ytr[perm], seed=52)
    shuffled, _ = accuracy(shuffled_head, xte, yte)
    print(json.dumps({
        "checkpoint": str(router.model_path),
        "train_count": len(train), "heldout_count": len(heldout),
        "neural_rule_priority_accuracy": round(score, 4),
        "untrained_head_accuracy": round(control, 4),
        "shuffled_labels_accuracy": round(shuffled, 4),
        "majority_baseline_accuracy": .5,
        "trained_head_changes_priority": sum(v != 0 for v in labels),
        "deployed": False,
        "caveats": [
            "The frozen encoder was originally trained for routing, not rule selection.",
            "Train and test share the same constrained grammar and may generalize poorly.",
            "A correct priority prediction does not prove fewer proof steps.",
            "No production model is modified by this experiment."
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
