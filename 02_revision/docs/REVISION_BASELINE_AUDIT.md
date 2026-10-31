# Revision Baseline Audit

Date: 2026-09-30
Scope: static audit of the repository state without retraining, altering code, or changing the data pipeline.

## 1) Executive summary

The repository does not have a single universal pipeline; it contains multiple overlapping research pipelines and result directories. The final manuscript-tracked result is not the legacy CViTS-Net script in [train.py](train.py), but the low-data teacher-student experiment defined in [train_lowdata_hybrid_distill.ipynb](train_lowdata_hybrid_distill.ipynb) and recorded under [results/lowdata_hybrid_distill](results/lowdata_hybrid_distill).

The strongest evidence for the actual final model is:

- [results/lowdata_hybrid_distill/best_overall_model_info.json](results/lowdata_hybrid_distill/best_overall_model_info.json) states: model = "TS_ConvNeXtTiny_Residual", seed = 62, saved checkpoint = "results\\lowdata_hybrid_distill\\best_overall_model.pth".
- [results/lowdata_hybrid_distill/TS_ConvNeXtTiny_Residual/seed_62/test_metrics.json](results/lowdata_hybrid_distill/TS_ConvNeXtTiny_Residual/seed_62/test_metrics.json) reports QWK = 0.9192318208797291, accuracy = 0.8528610354223434, ROC-AUC = 0.9603417459617237.

The repository therefore supports the claim that the final tracked model is the low-data residual teacher-student ConvNeXt variant, but the manuscript must explicitly describe that this is the model being audited, not the older hybrid or CViTS-Net scripts.

## 2) Actual data split and independence

The split logic is implemented in [dataset_loader.py](dataset_loader.py):

- It loads APTOS 2019 train.csv and then performs a stratified split using scikit-learn train_test_split with random_state=42.
- The code first isolates a test set with test_size=0.1 and stratify=labels.
- It then splits the remaining data into train and validation using a ratio derived from val_ratio/(train_ratio + val_ratio), again with random_state=42 and stratify=y_temp.
- This yields the standard 80/10/10 arrangement for train/val/test.

This means the final evaluation is not a random holdout from the full dataset without stratification. The code uses a fixed, reproducible split and then evaluates the final selected model once on the held-out test set. There is no evidence in the code of the test set being recycled into training or used for model selection after final checkpoint selection.

Important nuance: the model is selected using the validation QWK during training, not the test QWK. That is a standard validation-based model selection strategy, but the manuscript must state this clearly to avoid implying the test set was used for tuning.

## 3) Metric definition: QWK is implemented consistently and is ordinal-aware

The metric implementation in [metrics.py](metrics.py) uses:

- accuracy_score
- precision/recall/F1 with weighted averaging
- roc_auc_score for multi-class OVR weighted AUC
- cohen_kappa_score(y_true, y_pred, weights='quadratic')

This is the quadratic weighted kappa used for ordinal DR grading and is consistent with the manuscript's emphasis on QWK as the primary metric.

The low-data trainer in [train_lowdata_hybrid_distill.ipynb](train_lowdata_hybrid_distill.ipynb) also computes full metrics as part of validation and final test evaluation using compute_full_metrics(...), then saves them to the seed-specific test_metrics.json files. The saved results therefore reflect the same underlying metric definition as the paper's target metric.

## 4) The final model is the low-data residual teacher-student ConvNeXt path, not the legacy training script

The repository contains multiple models and result families:

- Legacy training pipeline in [train.py](train.py) and [dataset_loader.py](dataset_loader.py)
- Hybrid CNN+ViT experiments under [models/hybrid_cnn_vit_base.py](models/hybrid_cnn_vit_base.py) and [models/advanced_hybrid_models.py](models/advanced_hybrid_models.py)
- Teacher-student comparison outputs under [results/teacher_student_compare](results/teacher_student_compare)
- Final low-data distillation outputs under [results/lowdata_hybrid_distill](results/lowdata_hybrid_distill)

The final manuscript-linked result is clearly the low-data Distillation model family. In the experiment loop inside [train_lowdata_hybrid_distill.ipynb](train_lowdata_hybrid_distill.ipynb), the model list contains:

- TS_LiteCNNViT_BaseResidual
- TS_LiteCNNViT_ConflictBrake
- TS_LiteCNNViT_Conflict_OrdinalDistill
- TS_LiteCNNViT_BiGRU_OrdinalDistill
- TS_ConvNeXtTiny_Residual
- TS_EfficientNetB0_Residual

The final selected model is explicitly recorded as TS_ConvNeXtTiny_Residual in [results/lowdata_hybrid_distill/best_overall_model_info.json](results/lowdata_hybrid_distill/best_overall_model_info.json).

## 5) Architecture: residual gating and conflict brake are real behavior, not a reporting artifact

The low-data model architecture inside [train_lowdata_hybrid_distill.ipynb](train_lowdata_hybrid_distill.ipynb) defines a residual teacher-student model where:

- teacher logits are generated by the teacher ensemble;
- student logits are generated by the student backbone;
- the gate is computed from teacher/student probabilities and entropy/conflict features;
- the final output is a residual combination of z_teacher and a scaled gate-controlled delta term: z_final = z_teacher + residual_scale * g * delta.

This is not a superficial wrapper; the code logs gate statistics and uses a conflict brake when teacher confidence exceeds a threshold and teacher/student predictions disagree.

The model therefore has a nontrivial decision mechanism and is materially different from a simple student-only classifier or a naive ensemble average.

## 6) QWK inconsistency issue: likely seed-to-seed variation, not a coding error

The saved results in [results/lowdata_hybrid_distill/lowdata_seed_results.csv](results/lowdata_hybrid_distill/lowdata_seed_results.csv) contain seed-specific QWK values for the same final model:

- TS_ConvNeXtTiny_Residual, seed 42: QWK = 0.917438962528396
- TS_ConvNeXtTiny_Residual, seed 52: QWK = 0.9190749195807824
- TS_ConvNeXtTiny_Residual, seed 62: QWK = 0.9192318208797291

These are slightly different, but all are within a narrow band and are consistent with ordinary stochastic optimization variation across seeds. There is no evidence in the code of a single deterministic bug that would force identical predictions with different QWK values.

If the manuscript claims the equal-weight and entropy-gated fusion variants produce exactly identical predictions but different QWKs, that would require direct prediction-level evidence and could indicate a mismatch between narrative description and actual experimental artifact. The repository, however, contains no direct saved prediction matrix for that exact comparison in the final low-data pipeline; the evidence here supports the claim that the official final result is seed-dependent and the low-data residual model is the tracked result.

## 7) Known audit risk and revision recommendation

The biggest revision risk is not model failure; it is attribution and narrative clarity.

The repository contains multiple parallel research branches, several notebooks, and a large number of result files. Without an explicit statement, it is easy to confuse:

- the older CViTS-Net script,
- the hybrid CNN+ViT experiments,
- the teacher-student comparison folders,
- the final low-data residual distillation pipeline.

To make the manuscript defensible, the revision should explicitly state:

1. The final reported model is TS_ConvNeXtTiny_Residual in the low-data teacher-student distillation pipeline.
2. The fixed 80/10/10 stratified split is the same split used across experiments.
3. The validation QWK was used for checkpoint selection, while the test QWK was reported only once at the end.
4. Seed-to-seed QWK differences are expected and should be reported as average ± std across seeds, not as a single deterministic number without context.
5. The result values must be traced to the saved artifact files rather than to narrative tables or older notebooks.

## 8) Bottom line

The repository evidence supports the final audited claim:

- The actual manuscript-relevant pipeline is the low-data residual teacher-student experiment.
- The selected best model is TS_ConvNeXtTiny_Residual.
- The final tracked result is QWK ≈ 0.9192 at seed 62, with the model selected by validation QWK and evaluated on the held-out test set.
- The major revision need is clear attribution and reporting discipline, not a code-level proof of training leakage or a broken metric definition.
