# Papiti inquiry-inspired curriculum (experimental)

This is a **software experiment inspired by educational principles**, not an official
implementation, certification, or endorsement by Korea's Ministry of Education
or the International Baccalaureate.

## Mapping educational principles to speech-model research

| Inspiration | Experimental implementation |
| --- | --- |
| 2022 curriculum: concepts before rote recall | Train multiple original forms for a stable communicative intent, with distinct style cues |
| IB MYP: inquiry | Ask whether the model can produce grammatical output for withheld combinations rather than count remembered labels |
| Application / transfer | Add training-only Korean and English expressions using styles 4/5 while keeping earlier combinations 0..3 frozen |
| Reflection and process assessment | Log train/validation loss, compare against previous checkpoint, and hand-review grammaticality, meaning, and verified factual content |

This curriculum intentionally does **not** inject fixed answers into
the runtime. It provides extra *supervised training examples* only.

## Design controls

- The existing 94-example baseline validation/test combinations do not change.
- New learning examples are for nine social speech acts across Korean and English.
- New style cues 4 and 5 are added as training-only examples, not arbitrary
  runtime knowledge. Decoder can sample learned 4/5 combinations when opted in.
- Project identity and other factual content remain sourced from the existing
  semantic planner; the experiment does not invent or overwrite grounded facts.
- A lower training-copy rate is not sufficient for success. Repeated or broken
  non-copied text must fail manual quality review.
- The old 94 held-out combinations have been examined repeatedly; they are a
  regression benchmark, **not** a truly blind open-domain generalization test.
- No weights are trained by adding these files alone; users explicitly run
  the target below in their own environment.

## Run

```bash
make papiti-lm-train-v0.6.2-inquiry PAPITI_PYTHON=.venv-train/bin/python
.venv-train/bin/python -m papiti_lm.v0_6_1 eval \
  --checkpoint artifacts/papiti_language_v0_6_2_inquiry.pt \
  --report artifacts/papiti_v0_6_2_inquiry_eval.json
.venv-train/bin/python scripts/compare_prose_eval.py \
  artifacts/papiti_v0_6_2_repeat2_eval.json \
  artifacts/papiti_v0_6_2_inquiry_eval.json
.venv-train/bin/python scripts/review_prose_quality.py export \
  artifacts/papiti_v0_6_2_inquiry_eval.json \
  artifacts/papiti_v0_6_2_inquiry_review.csv
```

The baseline and inquiry runs use the same seed and validation/test splits.
They differ in additional training examples, so this is an **augmentation
ablation**, not a controlled measurement of educational efficacy.

## Release gate

First *provisional* benchmark: training reply reuse <=40%, grammatical
answers >=90%, semantic fidelity >=95%, serious fact errors 0. The latter
three require actual human review. Run the review script's `score` action
after reviewing all 94 rows. Add a separate genuinely unfamiliar dialogue
benchmark before any release recommendation.

Sources:
- Korean Ministry of Education, 2022 revised national curriculum:
  https://www.moe.go.kr/boardCnts/viewRenew.do?boardID=294&boardSeq=93459&lev=0&m=020402&opType=N&s=moe&statusYN=W
- International Baccalaureate, how the MYP works:
  https://ibo.org/programmes/middle-years-programme/what-is-the-myp/how-the-myp-works/
- International Baccalaureate, approach to teaching:
  https://www.ibo.org/benefits/the-ib-teaching-style/
