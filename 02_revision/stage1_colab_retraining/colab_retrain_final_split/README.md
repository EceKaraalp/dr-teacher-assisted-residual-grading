# APTOS DR manuscript-final-model split experiment

## Scope and honest interpretation

This is an isolated follow-up experiment. It does not overwrite the earlier `aptos_fullfactorial_20261011_v2` outputs. Its purpose is to test the manuscript's described final configuration using the manuscript's reported split sizes:

1. Teacher 1 and the equal-logit-fusion Teacher 2 system are retrained from the documented initialization and recipe.
2. The student is trained with the manuscript's equal-logit teacher/equal teacher-student inference configuration for seeds 42, 52, and 62.
3. The split has 2,929 training, 366 validation, and 367 test images. To retain a paired test set against the previous Colab run, the previous frozen test IDs are kept and exactly one sample is moved from validation to training. This does not recover the unknown historical IDs for the manuscript's original split.
4. The original checkpoints are unavailable; the experiment is a fresh retraining, not a replay from historical weights. Validation QWK chooses checkpoints; the test split is not used for early stopping.

Teacher models use seed 42 and are held fixed across the three student seeds. Therefore the three-seed student variation does not include teacher-training variation. The new split and the old split differ by exactly one train/validation sample while holding the test IDs fixed, but an identical old result is not guaranteed: the moved training sample can affect optimization, class weights, early stopping, and the fitted teacher/student parameters. Seed-level variation and missing historical checkpoints also prevent this from identifying the original weights.

â€œEqual-logit teacher/equal teacher-studentâ€ is the manuscript's described final fusion: Teacher 1/Teacher 2 logits are averaged, then teacher/student logits are averaged at inference. This package runs only `teacher_equallogit__student_equal`, not the gate-based sensitivity arms or augmentation ablation.

The purpose is to compare this freshly trained final configuration and split against the old Colab result and manuscript numbers. It does not establish external-hospital generalization.

## Required Google Drive layout

Upload:

```text
MyDrive/DR_Research/colab_retrain_final_split.zip
MyDrive/DR_Research/data/train.csv
MyDrive/DR_Research/data/cache_224.npz
```

`cache_224.npz` is approximately 551 MB. The package also supports rebuilding the cache from `train_images/`, but uploading the cache avoids transferring the much larger original image directory. The GitHub repository does not contain the dataset or model checkpoints. The code package can instead be pushed to GitHub and loaded with `USE_GIT=True` in the notebook.

Optional independent-site assessment:

```text
MyDrive/DR_Research/external/labels.csv   # id_code,diagnosis
MyDrive/DR_Research/external/images/     # filenames begin with id_code
```

Labels must use the APTOS ordinal mapping: `0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative`. Confirm that the external dataset's grading definition and image protocol are compatible before interpreting results.

## Run order

Open `Colab_Retrain_DR_FinalSplit.ipynb`, enable a GPU runtime, and run cells in order:

1. Mount Drive, unpack code, install `timm`, copy data locally, and verify GPU availability. Keep the same `RUN_ID` to resume from Drive; change it to isolate a new run.
2. Optionally run a stratified, one-epoch smoke test (`RUN_SMOKE = True`). Smoke results are not research results.
3. Verify image-cache dimensions, 80/10/10 split counts, and exact sample-ID agreement with the frozen split manifest.
4. Train Teacher 1 and the three Teacher 2 fusion-objective variants.
5. Train all configured student arms and seeds.
6. Run the validation/test analysis and write per-sample prediction evidence.
7. Generate Grad-CAM examples for the final gated seed-42 model.
8. Optionally run external-dataset evaluation (after adding external data); summarize results and sync to Drive.

Expected full training is compute-intensive and may take many hours or longer depending on Colab GPU availability. Checkpoints and results are mirrored periodically and after each completed model. If disconnected, restart the runtime, rerun setup, and rerun the interrupted stage. Training resumes from the last epoch checkpoint with saved Python, NumPy, CPU Torch, and CUDA random states. Identical results across GPU types/framework builds are not guaranteed.

For a shorter run, change `ARMS` or `SEEDS` in the setup cell only for debugging or explicitly scoped experiments; the resulting run is not the complete 11-arm/three-seed comparison. Keep all three teacher objectives in the teacher stage when running any student arm, since the runner validates all teacher checkpoints before student training.

## Expected outputs

Run ID: `aptos_manuscript_final_split366_20261012_v1`.

- Teachers: T1 and the three T2 fusion objectives are trained to satisfy the runner dependencies and preserve the existing pipeline; the target student arm uses the equal-logit-trained teacher.
- Students: one final-model arm, seeds 42/52/62.
- Split check: 2,929/366/367; test IDs must match the previous frozen test IDs; exactly one old validation ID must move to training.
- Analysis: teacher metrics, student mean/SD, per-seed outputs, split audit, duplicates and latency for the final arm.
- No augmentation arm is run here. Augmentation must not be inferred from this experiment.

## Metrics and artifacts

Under `MyDrive/DR_Research/results_retrain/<RUN_ID>/`:

- `teachers/` and `students/`: checkpoints, training histories, selected validation QWK, and test metrics.
- `split_ids.json`, `teachers/env_and_split.json`: IDs, counts, class balance, split protocol, class weights, and environment.
- `analysis/all_metrics.csv`: validation and test metrics for teacher, student branch, and full systems.
- `analysis/detailed_metrics.json`: confusion matrices and per-class precision, recall/sensitivity, F1, and support.
- `analysis/report.json`: paired comparisons and 95% bootstrap intervals, gate values, duplicate-image audit, augmentation and student-only comparisons, latency, parameter counts, and metric definitions.
- `analysis/duplicate_audit.csv`: image IDs, split, labels, duplicate-group ID, cross-split overlap and label conflicts.
- `analysis/predictions_*.csv` and `*.npz`: sample IDs, ground truth, probabilities, logits, and gates, preserving exact sample-level evidence.
- `analysis/student_summary_mean_sd.csv`: mean, standard deviation, and seed count for each student arm on the held-out test split.
- `gradcam/`: figure and selected case metadata.
- `external/`: external-dataset metrics and sample-level predictions, when run.

Primary metrics include QWK, accuracy, macro precision/recall/F1, weighted F1, macro and per-class specificity, weighted and per-class one-vs-rest ROC-AUC, mean absolute grade error, within-one-grade accuracy, multiclass Brier score, negative log-likelihood, 15-bin expected calibration error, confusion matrix, and class-wise scores. Paired bootstrap resamples within true class, pairs the two methods on the same images, uses 2,000 replicates, and reports percentile 95% intervals for QWK, accuracy, and macro F1. Undefined ROC-AUC is recorded as unavailable, not as zero.

## Important reporting limits

- The new run will not necessarily reproduce old manuscript metrics; use the newly generated results consistently throughout the revised manuscript and response letter.
- APTOS-only results do not answer the external-hospital validation request. Run the optional external evaluation only after obtaining appropriately labeled data.
- Three seeds provide a limited estimate of training variability. Report per-seed results and mean Â± standard deviation; do not present the paired image bootstrap interval as a confidence interval over independent training runs.
- The test set must remain a final evaluation set. Do not use test results to tune thresholds, select seeds, or choose the paper's model.
- Teachers are shared across student seeds; label the reported standard deviation as conditional student-seed variability, not end-to-end pipeline variability.
- Grad-CAM shows model sensitivity regions, not verified lesion segmentation or clinical causal evidence.

