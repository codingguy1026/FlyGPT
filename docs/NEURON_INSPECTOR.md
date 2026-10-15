# MaleCNS neuron-type inspector

FlyGPT's Brain Observatory provides a **verified biological annotation layer**
alongside the existing relative router activity trace.

## Data provenance and classification

The official Janelia **MaleCNS v1.0** release publishes curated body annotations
including `bodyId`, `superclass`, and `type`. FlyGPT queries the matched
`Neuron` objects in the neuPrint `male-cns:v1.0` dataset for the checkpoint's
actual body IDs. This uses a single, bounded batch query (maximum 512 IDs).
The UI fetches the annotations only when Brain Observatory is opened, and
the API caches a matching batch for up to 10 minutes.

Source: https://male-cns.janelia.org/download/
Scientific description of official superclass categories:
https://doi.org/10.1016/j.cell.2026.08.015

The viewer translates selected **exact** official `superclass` values into
coarse, *derived* display groups:

| Viewer | Official superclasses |
| --- | --- |
| Sensory | ol_sensory, cb_sensory, vnc_sensory, sensory_ascending, sensory_descending |
| Motor | cb_motor, vnc_motor |
| Intrinsic / interneuron | ol_intrinsic, cb_intrinsic, vnc_intrinsic |
| Projection / long-range | visual_projection, visual_centrifugal, ascending_neuron, descending_neuron |
| Other | any other nonempty superclass, including efferent, endocrine, TBC |
| Unclassified | no superclass or no neuPrint annotation |

Intrinsic is a broad explanatory grouping, **not** an experimentally proven
interneuron subtype for the FlyGPT routing task. Descending or ascending neurons
are not automatically motor neurons; non-motor efferents are not called motor.
A missing tag is NEVER filled with a guess based on graph structure, position,
NT prediction, or AI-generated text.

The inspector also displays raw neuPrint `type`, `instance`, and
`consensusNt` when present, leaving missing fields explicit. The scientific
annotation describes *fly biology*, not what role the 256-node FlyGPT task
router has learned for a node.

## UI

- Existing green activity-map mode stays unchanged and derives brightness
  only from recorded router activation frames.
- In the **뉴런 종류** tab, the graph recolors **only** using actual retrieved
  annotations. Categories are individually filterable and show counts.
- Selecting a neuron highlights its graph edges and shows its MaleCNS body ID,
  category, original superclass, type, instance, consensus NT, and relative
  router activity at the selected frame.
- When no neuPrint token or network service is available, classification
  stays unavailable/unclassified. The activity view remains operational and
  offers retry. A successful empty dataset never invents role information.
- Positions remain a *schematic topology layout*. Categories are not
  calcium imaging, functional imaging, or anatomical spatial coordinates.

## Codespaces smoke test

```bash
python -m unittest tests.test_neuron_annotations -v
cd frontend && npm run build
cd .. && make run
```

With a valid `NEUPRINT_TOKEN` on the private Codespace:

1. Open FlyGraph / Brain Observatory and switch to **뉴런 종류**.
2. Wait for "공식 주석 확인: N / 256개". Inspect an annotated and an
   unclassified neuron and verify raw superclass/type/NT values.
3. Switch back to **활성도**; verify replay still tracks the original
   trace and selected-node body ID remains the same.
4. Temporarily unset the token in a disposable test environment and confirm
   the UI shows an annotation error without fake classes.
5. Confirm an unauthenticated request to `/api/router/annotations`
   responds with HTTP 401 and does not query neuPrint.

Unit tests mock neuPrint and do **not** verify availability/latency or real
per-neuron annotations in a live Codespace. Keep ports and tokens private.
