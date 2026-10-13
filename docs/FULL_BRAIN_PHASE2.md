# Full MaleCNS: Phase 2, real structural signal propagation

This is an experimental **offline** brain-graph engine. It runs a weighted,
directed signal-propagation calculation across the full released weights table,
restricted to **curated annotated neuronal bodies**. It does not replace the
256-node trainable FlyGraphRouter or the Qwen language model. It is not wired
into \`/api/chat\`, and does not claim to model biological electrical firing.

The previously audited dataset had **151,856,684 connection rows**,
**311,833,243 synapse-weight total**, no nonpositive weights and no null rows.
Those figures include non-neuronal segments. Thus the next necessary input is
the official annotation file; do not treat all unique raw segment IDs as neurons.

Official data: https://male-cns.janelia.org/download/ (CC-BY)

## 1. Download the small curated neuron annotation file

The original 1.05 GB Feather weights file from phase 1 is reused unchanged.
The annotation Feather file is approximately 13 MB. Check space first, and
do not commit these data files to Git:

\`\`\`bash
df -h .
mkdir -p data/malecns_full
curl -fL --retry 3 -C - -o \
  data/malecns_full/body-annotations-male-cns-v1.0-minconf-0.5.feather \
  "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather"
\`\`\`

\`full_brain.py\` reads \`bodyId\` and \`superclass\` from the annotation table.
It keeps all nonempty neuronal superclasses, including TBC/untyped types,
and rejects explicit glia. The **actual eligible count is computed** from the
downloaded file, not hardcoded. No anonymous segment is secretly promoted
to a biological neuron.

## 2. Run cheap regression tests

\`\`\`bash
python -m unittest discover -s tests -p "test_full_brain.py" -v
\`\`\`

Fixtures cover multiple Feather batches, repeated directed edges, neuron
annotation filtering, self/zero-weight exclusion, the result guard, and
three-hop propagation.

## 3. First real whole-graph pass

Choose the official example neuron body ID 12781. Start with ONE step
to calibrate performance, rather than launching three entire-file scans at once:

\`\`\`bash
python full_brain.py 12781 --steps 1 --top 10
\`\`\`

The command scans **every record batch** of the 1.05 GB file and propagates
positive activation to all connected annotated neurons; no 256-node
restriction, top-edge sampling, or implicit synapse cutoff is applied.
\`min_weight=1\` is the default. Progress appears on stderr as batches advance.

Read these response fields:
- \`annotated_neurons_eligible\`: number of curated neuronal bodies from
  the downloaded annotation table
- \`stages[].rows_scanned\`: actual source-file connection rows examined
- \`stages[].contributing_connection_rows\`: eligible directed connections
  from currently active neuron sources
- \`stages[].active_neurons\`: all reachable neuronal bodies in the stage
- \`stages[].top\`: ten largest **normalized structural** activity scores

If the first pass is acceptable for the available machine, then try:

\`\`\`bash
python full_brain.py 12781 --steps 3 --top 10
\`\`\`

Each step independently rescans the full weights file. No compressed CSR
matrix is materialized yet; this deliberately keeps disk usage low but may
be slow for interactive use. Stop with Ctrl+C if resources become tight.
No operation mutates either official Feather file.

## 4. Exact interpretation

The update at each step is:

\`next[post] = sum(active[pre] * synaptic_weight(pre, post))\`

Values are rescaled by the **maximum positive value in each step**. The
scores are relative structural propagation, **not firing probabilities**,
membrane voltages, measured fly behavior, or learned natural-language
reasoning. All annotated neurons are eligible but only neurons reachable
from the selected source become active. This is a graph computation, not a
claim that every one of 166k neurons fires for every query.

Safety and correctness:
- A missing annotation file is a clear error, never a silent raw-segment
  fallback.
- If the node-cap guard is too small or exceeded, computation fails rather
  than truncating hidden signals.
- Tests use tiny synthetic data; running them is not a validation of the
  official 1.05GB source file or a guarantee about Codespaces performance.
- User-provided original model, training artifacts, and \`/api/chat\` stay
  untouched.

## Next step (not completed in this PR)

Benchmark the first full pass and build a persistent sparse/indexed graph
that supports repeatable, low-latency graph propagation with all eligible
neurons. Only after validating resource use and accuracy can the
full-connectome route be integrated into FlyGPT chat as an **opt-in** feature.
