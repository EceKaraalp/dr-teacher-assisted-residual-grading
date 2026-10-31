# APTOS DR Colab retraining and reviewer-evidence package

## Scope and honest interpretation

This package reruns the APTOS training pipeline without relying on the missing historical checkpoints:

1. Teacher 1 baseline is trained anew.
2. Teacher 2 (anchored SpectralNorm) is trained anew under three distinct fusion objectives: learned probability gate, equal-probability fusion, and equal-logit fusion. Each objective has its own checkpoint; Teacher 1 is the common frozen baseline.
3. Student models are trained with seeds 42, 52, and 62 for the complete, paired matrix:
   - Teacher fusion: learned probability gate, equal probability average, equal raw-logit average, Teacher 1 alone, or Teacher 2 alone.
   - Student fusion: learned residual gate or equal-logit average.
   - Includes augmentation arms for both the learned-teacher/learned-student-gate configuration and the manuscript's equal-logit teacher/equal-student configuration.
   - The default matrix is 36 student runs (12 arms x 3 seeds), in addition to Teacher 1 and all three Teacher 2 objective runs. The gate/equal-probability/equal-logit students use the Teacher 2 checkpoint trained under that same teacher objective. T1-only and T2-only are inference signal ablations using the gate-trained Teacher 2 checkpoint.
   - Equal-probability Teacher 2 training minimizes NLL on the mean of T1/T2 probabilities; equal-logit training minimizes cross-entropy on the mean raw logits. The learned-gate arm retains the gate entropy regularizer. This makes these teacher objectives actual retraining arms rather than post-hoc swaps on one gate-trained checkpoint.
   - All arms share the same split and optimization/loss recipe except the explicitly varied fusion objective, student fusion, or augmentation factor.
4. Validation QWK chooses the checkpoint. The test split is not used for early stopping or model selection.

Teacher 1 and each of the three Teacher 2 objective variants are trained once with seed 42 and then held fixed across the student-seed runs. Thus, the reported three-seed variability estimates student initialization/training variability conditional on the corresponding teacher checkpoint; it does not include teacher-training variability. Teacher single-run metrics are not a three-seed teacher estimate.

“Retrained anew” means no old DR checkpoint is loaded. The student ConvNeXt and Teacher 2 ResNet/ViT backbones use their specified ImageNet pretrained initialization; Teacher 1 is initialized by its architecture and trained on APTOS. This is not a claim that every parameter starts randomly initialized.

The outputs can resolve the gate-versus-equal description and support reviewer responses about augmentation, teacher contribution, inference cost, and explainability. They cannot establish performance on a different hospital dataset unless the user supplies a separately labeled external dataset. An optional evaluation-only script and notebook cell are provided for that; external data is not used for training or tuning.

## Required Google Drive layout

Upload:

```text
MyDrive/DR_Research/colab_retrain.zip
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

Open `Colab_Retrain_DR.ipynb`, enable a GPU runtime, and run cells in order:

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

## Experimental comparisons

- **Teacher-level ensemble:** reports standalone Teacher 1 and gate-trained Teacher 2; the learned probability gate, equal-probability, and equal-logit variants each use their separately trained Teacher 2 checkpoint. Student arms measure each signal's downstream contribution.
- **System-level gate/equal:** reports two different tests for each teacher signal:
  1. Inference-only swap on the exact same gated-trained student: learned residual gate versus arithmetic equal-logit average. The per-image predictions are compared directly; identical predictions must produce identical QWK.
  2. Separately retrained gated versus equal student for the same teacher signal, paired by seed and test image. This is the fair training-arm comparison for deciding what model the revised manuscript should describe.
- **Augmentation:** compares each augmented arm with its exact non-augmented counterpart by seed. This includes the manuscript-aligned equal-logit teacher/equal-student system (`teacher_equallogit__student_equal_aug`) and a separate gated-system sensitivity arm. Teacher models, split, optimizer recipe, and validation-only selection remain fixed; augmentation is applied only to student training images.
- **Teacher contribution:** standalone Teacher 1/2 metrics, their teacher-level fusion comparison, and downstream student retraining with the learned ensemble, either teacher alone, or fixed averages indicate whether the weaker teacher adds value. Since retraining the teacher ensemble itself alters its predictions, interpretation must separate ensemble-training and teacher-signal inference effects.
- **Runtime/system load:** times the trained student branch, teacher ensemble, and complete system at batch sizes 1 and 16; reports parameter counts. Measurements are forward-pass-only and exclude image loading/preprocessing.
- **Explainability:** Grad-CAM overlays target the predicted class score through the student's final ConvNeXt feature stage. Images are selected by true class for a balanced illustration; the figure is qualitative and is not a lesion-localization validation.
- **External generalization:** `external_eval.py` evaluates frozen APTOS-trained checkpoints on an independently labeled dataset without retraining or tuning. Results are only produced if the user supplies that dataset.

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
- Three seeds provide a limited estimate of training variability. Report per-seed results and mean ± standard deviation; do not present the paired image bootstrap interval as a confidence interval over independent training runs.
- The test set must remain a final evaluation set. Do not use test results to tune thresholds, select seeds, or choose the paper's model.
- Teachers are shared across student seeds; label the reported standard deviation as conditional student-seed variability, not end-to-end pipeline variability.
- Grad-CAM shows model sensitivity regions, not verified lesion segmentation or clinical causal evidence.
