# FlyGPT / Papiti v0.5: category-free full-MaleCNS policy experiment

Status: **experimental only, not deployed**. This phase demonstrates real
full-neuron policy learning; it is not yet the replacement for the production
chat routing, tool execution, or natural-language generation system.

## What changed
- **No fixed six-way route classification:** no `general/code/math/memory/
  research/summarize` softmax head. The v0.5 prototype predicts independent
  scores for concrete *available capability interfaces*: `web_search`,
  `memory_lookup`, and `calculator`. It may select any subset, including zero
  or multiple tools. Future tools are explicit interfaces, not topic categories.
- **Full 166,700-neuron population:** download the complete *official*
  Janelia MaleCNS v1.0 connectome and curated neuronal annotations. Pretrain
  separate directed-edge structural embeddings; require exactly **166,700**
  annotated neurons. Keep every neuron in the second-stage positive attention
  moments, including neurons lacking eligible positive edges. Each neuron has
  one independently trainable positive gain receiving gradients from policy
  supervision. No top-K truncation or 256-neuron fallback is accepted.
- **Language model contract (provisional):** output tool recommendations and
  constrained general instructions, with the original request preserved.
  The prototype does not silently execute tools or modify the existing chat
  route. Execution and factual answer verification still need integration.
- **Evidence:** training and 36 untouched tool-selection holdouts are separated.
  Exact action-set accuracy, per-tool F1 and unnecessary-tool rate are reported.
  A high score is not a demonstration of broad conversational intelligence.

## Run on GitHub Actions

Open the GitHub Actions workflow:
`.github/workflows/papiti-v0-5-full-malecns.yml`.

A pull request changing the v0.5 experiment files triggers it. The workflow:
1. Runs pure policy and split-integrity unit tests.
2. Downloads the official full ~1.1 GB source data (not a synthetic graph).
3. Pretrains structural keys from the complete curated directed connection set.
4. Checks the exact 166,700-neuron count and data provenance.
5. Trains the independent multi-label tool policy and individual neuron gains.
6. Tunes action thresholds using validation only and evaluates untouched holdout.
7. Uploads an experimental checkpoint and JSON metrics without deploying.

No user chat logs or credentials are committed. The original site and existing
checkpoint remain unchanged.

## Example intended policy

- `"사과와 바나나의 공통점이 뭐야?"` → no tools; respond directly.
- `"오늘 야구 경기 결과 알려줘"` → web search for current evidence.
- `"지난번에 정한 예산에서 2만원 빼 줘"` → memory + calculator.
- `"전에 정한 여행 예산을 오늘 환율로 계산해 줘"` → memory + web + calculator.

Those are **targets in a labeled training task**, not guaranteed predictions.
Review independent holdout results before considering integration.

## Scientific limitations

This model is not a biophysical simulation or a live flight-brain thought
process. Most neuron embeddings learn from official directed positive edges,
but annotations currently include 217 neurons without such eligible edges.
Those neurons are retained and receive *policy-stage gain gradients*, though
this cannot recover missing anatomical knowledge. The graph is represented
through exact full-population moments rather than full real-time synaptic
propagation.

An ordinary hash-feature text encoder and 128 labeled examples are not enough
to assert general language comprehension; a pretrained multilingual semantic
encoder, larger diverse human-reviewed instruction/tool corpus, blind external
evaluations, randomized-connectome and no-graph ablations, policy calibration,
and a safe execution layer are needed for production.

The existing `papiti-language-v0.6.1` repeated self-introduction issue is a
separate response-generation/knowledge-selection bug, not resolved by this
routing-policy experiment. Never conflate the tool-selection benchmark with
end-to-end answer quality.
