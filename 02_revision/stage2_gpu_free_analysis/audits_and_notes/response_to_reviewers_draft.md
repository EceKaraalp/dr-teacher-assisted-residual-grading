# Response to Reviewers – DRAFT (not for submission)

Only statements that are currently true are written as done. `[PENDING]` = experiment not yet run; do not submit until filled.

## Reviewer 1

**R1.1** Thank you. We re-examined the stored results. For seed 42 the Gate2 and equal-weight rows give identical accuracy, F1 and QWK (0.9174); only AUC differs (0.958 vs 0.945). `[PENDING: per-sample predictions to demonstrate identity and recompute with one metric implementation.]`

**R1.2** We agree. Candidate student families were ranked using test QWK; epoch selection within runs used validation QWK. We now state this in Limitations (revised manuscript, last paragraph of Limitations) and no longer describe the test set as fully independent. External validation `[PENDING]`.

**R1.3** `[PENDING]` Student-only and full-system results will be reported separately.

**R1.4 / R1.6 / R1.7 / R1.8 / R1.9 / R1.10 / R1.16** `[PENDING]` controlled ablations, teacher contribution, aggregation rule, calibration (ECE/Brier/NLL), normalization, data-fraction study, and Fig. 2 for both inference modes.

**R1.5** We will remove superiority wording relative to the standalone CNN–ViT fusion model (QWK 0.923 vs 0.9186) and describe the benefit as compactness. `[PENDING text edit]`

**R1.11** `[PENDING]` loss definition to be rewritten with exact formulas and hyper-parameters.

**R1.12** The split manifest (2,928/367/367 with image IDs, seed 42), hyper-parameters and seeds will be released. Checkpoint release `[PENDING]`.

**R1.13** We found the manuscript parameter total (~149.8M) inconsistent with the code count (141,999,859 total; 27,824,454 trainable) and will correct it `[PENDING]`; latency will be re-measured on one machine `[PENDING]`.

**R1.14 / R1.15** Added class-wise recall/F1 and referable-DR (grade ≥ 2) sensitivity 0.964 ± 0.004 and specificity 0.947 ± 0.003 (three seeds) with confusion-matrix-resampled 95% intervals (Section 4, after Fig. confusion matrix). Paired bootstrap/significance tests `[PENDING]`.

**R1.17–R1.21** `[PENDING]` literature comparison protocol statement, bibliography verification, clinical-claim toning, AI-use statement, formatting.

## Reviewer 2

**R2.1** `[PENDING]` external dataset evaluation; the limitation is stated in the manuscript.
**R2.2** `[PENDING]` augmentation ablation.
**R2.3** We observed that Teacher 1 results vary across our runs (test QWK 0.64–0.72), and will justify its inclusion only after the Teacher-2-only vs two-teacher comparison `[PENDING]`.
**R2.4** `[PENDING]` student vs full-system latency/accuracy trade-off.
**R2.5** `[PENDING]` Grad-CAM heat-maps (no such figures currently exist).
