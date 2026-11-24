# Papiti v0.6.5: IB-MYP-inspired training experiment

**Research analogy only. Not an official IB curriculum, IB-authorized assessment, or IB-endorsed model.**

Official background:
- https://www.ibo.org/benefits/the-ib-teaching-style/
- https://www.ibo.org/programmes/middle-years-programme/what-is-the-myp/how-the-myp-works/
- https://www.ibo.org/programmes/middle-years-programme/curriculum/language-and-literature/

## Learning stages (36 epoch setting)

1. **Inquiry (1-6):** teach opening and follow-up clauses independently using masked labels.
2. **Action (7-12):** transfer training-only opening and follow-up fragments into new mixed-style pairs. Each full-response epoch replaces up to 20% of unique original examples, rather than increasing its batch count.
3. **Reflection (13+):** inspect the teacher-forced loss of TRAINING-ONLY action probes once every three epochs and rehearse difficult examples within the same practice budget. This is not human-style reflection or semantic self-critique.

## Integrity

- Original validation and 94 fixed tests remain unchanged.
- Previously observed 36 pairs (4,0)/(0,5) and the second diagnostic 36 pairs (4,1)/(1,5) are excluded from optimization.
- Action pairs: (2,4), (3,5), (5,2), (5,3). Explicitly skip exact held-out answers.
- Both IB and control use the same original-training-only tokenizer, model capacity, unique training examples and number of training examples per full-response epoch. Only the IB run substitutes action examples and reviews training-side losses.
- Best epoch is chosen using validation char NLL with patience 7. There is no expectation that 36 epochs must be reached.
- No runtime answer tables, fixed responses or forbidden-word filters.
- Both earlier 36-pair test sets have been inspected repeatedly, so improvements on them should be considered exploratory. Preregister a truly unseen new test.

## Run in Codespaces

Run the guardrail tests:

    .venv-train/bin/python -m unittest tests.test_papiti_v0_6_5_ib_curriculum

Train and evaluate the IB experiment:

    make papiti-lm-train-v0.6.5-ib PAPITI_PYTHON=.venv-train/bin/python
    make papiti-lm-transfer-v0.6.5-ib PAPITI_PYTHON=.venv-train/bin/python

Optional matched control:

    make papiti-lm-train-v0.6.5-ib-control PAPITI_PYTHON=.venv-train/bin/python
    make papiti-lm-transfer-v0.6.5-ib-control PAPITI_PYTHON=.venv-train/bin/python

All outputs use new v0.6.5 artifact paths, leaving the v0.6.4 best checkpoint unchanged.
