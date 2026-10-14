# Experimental full-MaleCNS structural pretraining

The existing **256-neuron task router** remains unchanged. This is a separate,
offline, self-supervised **directed connection reconstruction** trainer for all
curated annotated MaleCNS v1.0 neuronal bodies. It reads every record batch of
the official Feather weights file and uses **every positive connection between
annotated neurons**. No 256-node selection, edge cap, or hidden top-K cutoff.

The number of eligible neurons is calculated from the official annotations
(including curated untyped classes); unannotated segments and explicit glia
are excluded. The saved checkpoint distinguishes eligible bodies from the
subset with observed positive annotated connections, because isolated neurons
cannot receive graph-based training signal.

## Run

Reuse the official weights and annotation downloads described in
[FULL_BRAIN_PHASE2.md](FULL_BRAIN_PHASE2.md). Confirm sufficient free disk
space for the source file (about 1.05 GB). The data files must NOT be committed.

```bash
df -h .
python -m pip install -r requirements-full-connectome.txt
# PyTorch + numpy must also be available (e.g. via make train-setup).
python -m unittest discover -s tests -p 'test_full_connectome_training.py' -v
python training/train_full_connectome.py --epochs 1 --batch-size 32768 \
  --out artifacts/malecns_full_structural_embeddings.pt
```

One epoch means a full pass through the huge file and can be compute-intensive.
The checkpoint contains sorted neuron body IDs, trainable pre/post structural
embeddings, scanned-row counts, eligible-edge counts, connected-neuron counts,
and epoch training loss. Negative destinations are sampled approximately;
some negatives may also be real connected partners. No held-out validation
has been demonstrated yet.

## Important limitations

This **does not** teach 160k neurons to speak, calculate, or learn human
etiquette. It is structural link reconstruction, not language supervision,
a biological firing simulation, or a verified high-accuracy classifier.
No full-data training has been run or checkpoint committed by this PR.
It is not integrated into /api/chat, and it should not replace the
256-neuron checkpoint without explicit evaluation and validation.

The separate chat regression tests guard Korean greetings, deterministic
math results, and internal plan leakage. For true all-neuron task-routing
training, the next step is a labeled graph-to-route readout with validation
against the original 256-node model.
