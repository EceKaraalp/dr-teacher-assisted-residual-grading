# Revision Critical Audit

Date: 2026-09-30
Scope: forensic audit of the unresolved reviewer issues in the diabetic retinopathy teacher-student pipeline, without retraining or altering any existing source, checkpoints, or outputs.

## 1) Executive conclusion

The repository contains a real and serious attribution problem: the numbers, model names, and narrative explanations do not always refer to the same experiment. The strongest evidence supports a single final manuscript-tracked path, but some reviewer-facing claims remain unprovable from the saved artifacts because the underlying prediction arrays are not present.

The most important unresolved contradiction is this:

- A narrative report in [DENEY_SONUCLARI_KISA_RAPOR.md](DENEY_SONUCLARI_KISA_RAPOR.md) states that Gate2 and Equal Weight produce identical predictions on the test set: 368/368 identical, with the same accuracy, F1, and QWK.
- The saved comparison CSV at [results/teacher_student_compare/teacher_student_compare_val_test.csv](results/teacher_student_compare/teacher_student_compare_val_test.csv) reports different test QWK values for the teacher-only ensemble comparison: Teacher_EqualWeight_2T,test = 0.9016402229845626 and Teacher_Gated_Ensemble,test = 0.906792740344228.
- The final audited model under [results/lowdata_hybrid_distill/TS_ConvNeXtTiny_Residual/seed_62/test_metrics.json](results/lowdata_hybrid_distill/TS_ConvNeXtTiny_Residual/seed_62/test_metrics.json) is a different model family from the earlier teacher-only comparison and reports QWK = 0.9192318208797291.

These are not the same experiment, and the repository does not contain the direct prediction arrays needed to prove or falsify the claim of identical predictions on the same sample set. The responsible revision position is therefore: the contradiction is real at the narrative level, but it cannot be fully resolved from the current saved artifacts.

## 2) Exact data split and counts

The split logic is implemented in [dataset_loader.py](dataset_loader.py) and is deterministic:

- first split: train_test_split with test_size = 0.1, random_state = 42, stratify = labels
- second split: split the remaining data into train and validation using val_ratio / (train_ratio + val_ratio), again with random_state = 42 and stratify = y_temp

This yields the standard 80/10/10 split, and the code prints the actual counts:

- Train: 2929
- Validation: 366
- Test: 367

The final saved metrics in [results/lowdata_hybrid_distill/TS_ConvNeXtTiny_Residual/seed_62/test_metrics.json](results/lowdata_hybrid_distill/TS_ConvNeXtTiny_Residual/seed_62/test_metrics.json) also show 367 samples in the confusion matrix and classification report (macro avg support = 367.0).

This conflicts with the narrative in [DENEY_SONUCLARI_KISA_RAPOR.md](DENEY_SONUCLARI_KISA_RAPOR.md), which says the test set contains 368 samples. The repository therefore mixes at least three different reporting references:

1. the code-level 80/10/10 split with 367 test samples,
2. the textual report with 368 test samples,
3. the saved comparison CSV and JSON artifacts with different model families and metrics.

## 3) The strongest evidence for the actual final model

The final manuscript-linked path is not the legacy CViTS-Net script, but the low-data teacher-student distillation pipeline in [train_lowdata_hybrid_distill.ipynb](train_lowdata_hybrid_distill.ipynb).

The repository states in [results/lowdata_hybrid_distill/best_overall_model_info.json](results/lowdata_hybrid_distill/best_overall_model_info.json):

- model = "TS_ConvNeXtTiny_Residual"
- seed = 62
- source checkpoint = results\lowdata_hybrid_distill\TS_ConvNeXtTiny_Residual\seed_62\best_model.pth

The same model reports the following final test metrics in [results/lowdata_hybrid_distill/TS_ConvNeXtTiny_Residual/seed_62/test_metrics.json](results/lowdata_hybrid_distill/TS_ConvNeXtTiny_Residual/seed_62/test_metrics.json):

- accuracy = 0.8528610354223434
- qwk = 0.9192318208797291
- roc_auc = 0.9603417459617237
- best_val_qwk = 0.888870675072343

This is the strongest evidence that the final tracked model is TS_ConvNeXtTiny_Residual, selected by validation performance and evaluated once on the untouched test set.

## 4) Validation-based model selection is supported by the saved artifacts

The final low-data metrics file contains both validation and test metrics:

- best_val_qwk = 0.888870675072343
- test qwk = 0.9192318208797291

This indicates the code selected the checkpoint based on validation performance, then reported the final test metric after the selection step. That is consistent with a standard train/val/test protocol, not a leakage-prone setup, provided the manuscript explicitly says the validation set was used for checkpoint selection and the test set was used only once for final reporting.

The manuscript should therefore say: model selection was based on validation QWK, while the final reported QWK comes from the held-out test set.

## 5) The teacher–student fusion logic is real but the inference mechanics are not uniformly documented

The repository’s baseline audit in [REVISION_BASELINE_AUDIT.md](REVISION_BASELINE_AUDIT.md) clearly describes the architecture as a residual teacher-student system in which:

- teacher logits are produced by the teacher ensemble,
- student logits are produced by the student backbone,
- a gate is computed from probabilistic or entropy/conflict information,
- the final output is a residual combination of teacher logits and a gate-controlled delta.

This means the code path is a real fusion mechanism, not a superficial wrapper. The design is materially different from simple averaging. However, the exact formulation and branch-specific behavior are not consistently explained across the narrative reports; some sections describe gate values close to zero, others describe dramatically different teacher/student contribution percentages, and some artifacts refer to teacher-only vs teacher+student settings without clear experiment alignment.

## 6) The contradiction around identical predictions vs different QWK values is not resolvable from the current artifacts

This is the critical issue for the reviewer response.

### Evidence of the contradiction

In [DENEY_SONUCLARI_KISA_RAPOR.md](DENEY_SONUCLARI_KISA_RAPOR.md), the narrative says:

- Test set: 368 samples
- Gate2 predictions = Equal Weight predictions: 368/368 (100%)
- Same accuracy, F1, and QWK

The same report also explains a gate value near 0.09, a conflict brake, and a claim that the student effectively contributes less than 10% while the teacher dominates the output. This narrative is internally consistent if the predictions truly match.

### Counter-evidence from saved comparison artifacts

The saved comparison CSV at [results/teacher_student_compare/teacher_student_compare_val_test.csv](results/teacher_student_compare/teacher_student_compare_val_test.csv) contains:

- Teacher_EqualWeight_2T,test = 0.9016402229845626
- Teacher_Gated_Ensemble,test = 0.906792740344228

These are different QWK values, and the model names are teacher-only ensemble variants, not the final TS_ConvNeXtTiny_Residual low-data model. The CSV therefore shows a totally different experiment namespace from the low-data final model.

### Interpretation

The most defensible conclusion is:

- the contradiction is real in the narrative layer,
- the repository does not provide the exact prediction-array artifacts needed to prove the label-level equality claim for the same experiment,
- the saved numeric artifacts point to different experiments and different runs,
- therefore the claim cannot be validated or disproven by the current repository state alone.

This is not a code bug in the metric definition; it is a documentation and artifact-traceability problem.

## 7) Low-data claim: the wording is risky unless the experiment is explicitly defined

The final pipeline is named a low-data teacher-student distillation workflow in [train_lowdata_hybrid_distill.ipynb](train_lowdata_hybrid_distill.ipynb), but the actual split logic in [dataset_loader.py](dataset_loader.py) still uses the standard stratified 80/10/10 split for train/val/test. The phrase “low-data” therefore appears to describe the experimental family or training regime, not a different data-split ratio.

Reviewer risk:

- if the manuscript claims a low-data setting, it must define what low-data means in this repository,
- if it claims “standard APTOS 2019 80/10/10 split,” it must not conflate that with a separate low-data experimental design,
- if it compares low-data models to teacher-only baselines, it must clearly separate those experiment families in tables and text.

## 8) Reviewer-facing recommendations

The revision should be written in a way that is both evidence-based and conservative:

1. State the actual final model explicitly: TS_ConvNeXtTiny_Residual in the low-data teacher-student distillation pipeline.
2. State the actual split protocol explicitly: 80/10/10 stratified split with random_state = 42; train 2929, val 366, test 367.
3. State the selection protocol explicitly: checkpoint selected on validation QWK; held-out test set not used for model selection.
4. Separate the experiment families cleanly: teacher-only ensemble comparison, low-data distillation pipeline, and final selected model.
5. Remove or qualify any statement that claims identical predictions on a test set unless the exact prediction arrays for that experiment are saved and compared.
6. Do not compare narrative claims from [DENEY_SONUCLARI_KISA_RAPOR.md](DENEY_SONUCLARI_KISA_RAPOR.md) directly to saved final metrics unless the two artifacts are explicitly tied to the same run, same model, and same test split.

## 9) Final judgment

The repository supports the following claims with strong artifact evidence:

- the final manuscript-linked model is TS_ConvNeXtTiny_Residual,
- the final reported test QWK is approximately 0.9192,
- the split is an 80/10/10 stratified split,
- validation QWK is used for checkpoint selection.

The repository does not support the following claim in a fully traceable way:

- that Gate2 and Equal Weight produce identical predictions on the same test set while also having different QWK values, when the actual prediction arrays are absent.

Therefore, the critical audit conclusion is: the contradiction is a documentation and artifact-alignment problem, not a verified numerical result from a single reproducible experiment.
