# Manuscript-final-model split experiment

## Purpose

This is a separate follow-up to `aptos_fullfactorial_20261011_v2`. It retrains the manuscript-described final system with the reported split counts and preserves the previous frozen test IDs for a paired comparison. The historical model checkpoints and exact historical split IDs are unavailable; this run can test whether the changed split count affects results, but cannot recover the original weights or guarantee exact reproduction.

## Prespecified configuration

- Split sizes: train 2,929 / validation 366 / test 367.
- Split adjustment: reproduce the previous two-step stratified split, then move one validation sample from the most frequent validation class into training. The held-out test IDs remain unchanged. The moved ID is recorded in `analysis/report.json`.
- Model: Teacher 1 baseline + Teacher 2 SpectralNorm hybrid trained with equal-logit teacher fusion; ConvNeXt Tiny student.
- Final fusion: equal-weight average of T1/T2 logits for the teacher, then equal-weight average of teacher/student logits at inference.
- Student arm: `teacher_equallogit__student_equal`.
- Student seeds: 42, 52, and 62. Teacher training uses seed 42.
- No augmentation, matching the manuscript's primary training protocol.
- Checkpoint selection: validation QWK. Test set is not used for checkpoint selection.
- Run ID: `aptos_manuscript_final_split366_20261012_v1`.

The package trains the shared T1 and all three T2 fusion-objective checkpoints because the existing pipeline's teacher-analysis stage expects them. Only the equal-logit teacher/equal-student student arm is trained.

## Comparison target from the previous Colab run

The same student arm on the prior split (2,928/367/367) had:

| Metric | Previous split, 3 student seeds |
|---|---:|
| Accuracy | 0.8474 Â± 0.0109 |
| Macro-F1 | 0.6947 Â± 0.0092 |
| QWK | 0.9014 Â± 0.0078 |
| ROC-AUC | 0.9696 Â± 0.0026 |

These are not the historical manuscript results (reported accuracy 0.8547 and QWK 0.9186); they are the previous Colab rerun and provide the controlled comparison for the one-sample split adjustment.

## Results

**Pending Colab execution.** After completion, transcribe `analysis/student_summary_mean_sd.csv`, `analysis/report.json`, and the three `students/teacher_equallogit__student_equal/seed_*/test_metrics.json` files here. Compare the new results against both the prior Colab row above and the manuscript's reported values. Do not claim the split adjustment caused any difference without also considering student seed variability and the single-seed teacher training.

## Limitations

- The exact historical sample IDs for the manuscript's 2,929/366/367 split are not available. The split here preserves the previous test IDs but may not be the manuscript's original split.
- One changed training/validation sample can affect fitted weights, class weights, early stopping, and validation-based checkpoint selection. A small count change does not guarantee a small or predictable metric change.
- The teacher models use one seed; the student uses three.
- This experiment does not address augmentation, external-dataset validation, or Grad-CAM.

