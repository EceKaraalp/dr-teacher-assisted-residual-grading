# MODEL_SELECTION_FORENSIC_AUDIT

Date: 2026-09-30
Scope: final forensic audit of the selection path and test-set independence for the final TS_ConvNeXtTiny_Residual model, without retraining or modifying repository artifacts.

## 1) FINAL MODEL SELECTION

The audited final path is the low-data teacher-student distillation workflow in [train_lowdata_hybrid_distill.ipynb](train_lowdata_hybrid_distill.ipynb). The model family comparison is performed inside that notebook during the screening/final experiment loop. The notebook prints one line per model, for example: Screening Experiment: TS_LiteCNNViT_BaseResidual, Screening Experiment: TS_LiteCNNViT_ConflictBrake, Screening Experiment: TS_LiteCNNViT_Conflict_OrdinalDistill, Screening Experiment: TS_LiteCNNViT_BiGRU_OrdinalDistill, Screening Experiment: TS_ConvNeXtTiny_Residual, and Screening Experiment: TS_EfficientNetB0_Residual. The saved result rows are in [results/lowdata_hybrid_distill/lowdata_seed_results.csv](results/lowdata_hybrid_distill/lowdata_seed_results.csv), and the final aggregate summary is in [results/lowdata_hybrid_distill/lowdata_summary_mean_std.csv](results/lowdata_hybrid_distill/lowdata_summary_mean_std.csv).

The final selection artifact is [results/lowdata_hybrid_distill/best_overall_model_info.json](results/lowdata_hybrid_distill/best_overall_model_info.json), which records:

- model: TS_ConvNeXtTiny_Residual
- seed: 62
- qwk: 0.9192318208797291
- saved_checkpoint: results\lowdata_hybrid_distill\best_overall_model.pth

Model-by-model evidence:

| Model | Validation QWK | Test QWK | Validation Accuracy | Test Accuracy | Seed(s) | Source result file | Timing |
|---|---:|---:|---:|---:|---|---|---|
| TS_LiteCNNViT_BaseResidual | 0.8859058000997511 | 0.9085044219145009 | not explicitly listed as Accuracy_mean; screen row has validation-linked metric in Best_Val_QWK = 0.8859058000997511 | 0.8501362397820164 | 42 | [results/lowdata_hybrid_distill/lowdata_seed_results.csv](results/lowdata_hybrid_distill/lowdata_seed_results.csv) | Available during screening; before final best-model selection |
| TS_LiteCNNViT_ConflictBrake | 0.8918074105746402 | 0.9116784807911088 | not explicitly listed as Accuracy_mean; screen row has validation-linked metric in Best_Val_QWK = 0.8918074105746402 | 0.8528610354223434 | 42 | [results/lowdata_hybrid_distill/lowdata_seed_results.csv](results/lowdata_hybrid_distill/lowdata_seed_results.csv) | Available during screening; before final best-model selection |
| TS_LiteCNNViT_Conflict_OrdinalDistill | 0.889791531665376 (seed 42), 0.8927855531285674 (seed 52), 0.8893242401023994 (seed 62) | 0.9151003879803212 (seed 42), 0.9136909013782993 (seed 52), 0.9175886341063177 (seed 62) | not explicitly listed as Accuracy_mean; best validation accuracy is not stored in the saved summary; only Best_Val_QWK is available | 0.8446866485013624 (seed 42), 0.8392370572207084 (seed 52), 0.8528610354223434 (seed 62) | 42, 52, 62 | [results/lowdata_hybrid_distill/lowdata_seed_results.csv](results/lowdata_hybrid_distill/lowdata_seed_results.csv) | Available during screening/final loop; before final selection |
| TS_LiteCNNViT_BiGRU_OrdinalDistill | 0.8901784586193673 | 0.9123138542806999 | not explicitly listed as Accuracy_mean; screen row has validation-linked metric in Best_Val_QWK = 0.8901784586193673 | 0.8446866485013624 | 42 | [results/lowdata_hybrid_distill/lowdata_seed_results.csv](results/lowdata_hybrid_distill/lowdata_seed_results.csv) | Available during screening; before final best-model selection |
| TS_ConvNeXtTiny_Residual | 0.89756861321322 (seed 42), 0.8859058000997511 (seed 52), 0.888870675072343 (seed 62); mean across seeds = 0.890781696128438 | 0.917438962528396 (seed 42), 0.9190749195807824 (seed 52), 0.9192318208797291 (seed 62); mean across seeds = 0.9185819009963024 | 0.8528610354223434 (seed 42), 0.8583106267029973 (seed 52), 0.8528610354223434 (seed 62); mean = 0.854677565849228 | same as test accuracy above (final rows in seed CSV) | 42, 52, 62 | [results/lowdata_hybrid_distill/lowdata_seed_results.csv](results/lowdata_hybrid_distill/lowdata_seed_results.csv) and [results/lowdata_hybrid_distill/lowdata_summary_mean_std.csv](results/lowdata_hybrid_distill/lowdata_summary_mean_std.csv) | Final model selected from summary ranking; final single-seed test file created after selection |
| TS_EfficientNetB0_Residual | 0.8795800065623974 | 0.9070946437892464 | not explicitly listed as Accuracy_mean; screen row has validation-linked metric in Best_Val_QWK = 0.8795800065623974 | 0.8446866485013624 | 42 | [results/lowdata_hybrid_distill/lowdata_seed_results.csv](results/lowdata_hybrid_distill/lowdata_seed_results.csv) | Available during screening; before final best-model selection |

Important: the notebook note states that the metrics in these rows are reported on the test set, while Best_Val_QWK is the validation metric. This is visible in the notebook text near the run notes and is consistent with the saved columns: Accuracy, QWK, ROC_AUC, and related metrics are test-set metrics; Best_Val_QWK is the validation-side checkpoint selection metric.

## 2) DETERMINING THE ACTUAL SELECTION RULE

The exact selection logic is in the summary-ranking block in [train_lowdata_hybrid_distill.ipynb](train_lowdata_hybrid_distill.ipynb). The notebook creates a summary table from the experimental results and sorts it by QWK_mean. The evidence is explicit in the saved code path:

- s = s.sort_values('QWK_mean', ascending=False).reset_index(drop=True)
- best_model_name = summary_df.iloc[0]['Model']
- print('\nBest model by QWK_mean:', best_model_name)
- best_info_path = results_dir / 'best_overall_model_info.json'

This means the actual selection rule is:

A. validation QWK
B. test QWK
C. validation accuracy
D. test accuracy
E. another metric
F. manual selection
G. cannot be determined

Answer: E. another metric, specifically the model ranking by QWK_mean from the aggregated seed summary, not by a validation-QWK criterion. The notebook does not select by test QWK directly; it ranks the models by the aggregate mean test QWK from the seed summary table and writes that result to the overall-best file.

The ranking is performed on the summary table built from the saved seed results, not on the raw validation split. The best model file therefore reflects summary-level ranking by QWK_mean, not checkpoint selection by validation score, and the final single seed used in the best-overall file is then selected from that winner.

## 3) TEST-SET USE DURING MODEL SELECTION

The repository does use test metrics in the selection pipeline, but not in the same way as a training-loop gradient step. The proof is not that the test set was absent from optimization; it is that the model-ranking stage explicitly sorts candidate models by QWK_mean using aggregated test-set metrics.

Repository evidence:

- [train_lowdata_hybrid_distill.ipynb](train_lowdata_hybrid_distill.ipynb): the notebook note says: “Not: Tum metrikler test seti uzerinden raporlanir; veri bolunmesi orijinal 80-10-10'dur.” (All metrics are reported on the test set; the original split is 80-10-10.)
- [train_lowdata_hybrid_distill.ipynb](train_lowdata_hybrid_distill.ipynb): the summary table is sorted by QWK_mean, which is created from the seed-level test metrics.
- [train_lowdata_hybrid_distill.ipynb](train_lowdata_hybrid_distill.ipynb): the selected model file is written from the ranked summary table, not from a held-out validation metric alone.
- [results/lowdata_hybrid_distill/lowdata_summary_mean_std.csv](results/lowdata_hybrid_distill/lowdata_summary_mean_std.csv): this file is a model-level aggregate of test-set results, with columns including Accuracy_mean, ROC_AUC_mean, QWK_mean, and Best_Val_QWK_mean.

This matters because the reviewer issue is not “test set was used in gradients”; it is “test-set metrics were used in model ranking.” The repository artifacts support the second statement for the summary-ranking stage.

## 4) SEED SELECTION

The audited seed comparison for TS_ConvNeXtTiny_Residual is:

- seed 42: QWK = 0.917438962528396
- seed 52: QWK = 0.9190749195807824
- seed 62: QWK = 0.9192318208797291

The summary file [results/lowdata_hybrid_distill/lowdata_summary_mean_std.csv](results/lowdata_hybrid_distill/lowdata_summary_mean_std.csv) reports the aggregate mean and standard deviation across these seeds:

- QWK_mean = 0.9185819009963024
- QWK_std = 0.0009929178014660813

The chosen single seed saved in the overall-best file is seed 62, but the notebook selection logic does not show that seed 62 was chosen because it had the highest single-run test QWK. In fact, seed 62 does have the highest single-run test QWK among the three saved values, but the model-ranking code selects the model by QWK_mean across seeds, and the final single seed is then chosen as the best seed within that model. The repository does not provide a separate stage saying: “seed 62 was chosen only because it had the highest test QWK and therefore overrides all other seeds.” The safer reading is:

- the model family was chosen by the aggregate QWK_mean ranking,
- the specific single seed for that model was selected from the best-seed record within the model group,
- the final file therefore reflects a model-level aggregate ranking plus a per-model best-seed choice.

This means the underlying issue is not cleanly “test-set-free” model selection; it is a summary-level test-metric ranking with a chosen best single seed inside the winner.

## 5) BEST OVERALL MODEL INFO

The file [results/lowdata_hybrid_distill/best_overall_model_info.json](results/lowdata_hybrid_distill/best_overall_model_info.json) contains only the winner after the ranking stage. It includes:

- model
- seed
- qwk
- source_checkpoint
- saved_checkpoint

It does not contain a validation metric field. It stores a single winner after ranking, rather than a validation-set selection record. The file is generated in the summary-selection block in [train_lowdata_hybrid_distill.ipynb](train_lowdata_hybrid_distill.ipynb), immediately after the model summary is sorted by QWK_mean and the top row is selected. The actual selection logic can be traced to those lines. The file is therefore written after the experiment summary and ranking stage, not before it.

## 6) RESULT TABLE TRACEABILITY

Manuscript number: QWK = 0.9186 ± 0.0010

Exact repository source:

- [results/lowdata_hybrid_distill/lowdata_summary_mean_std.csv](results/lowdata_hybrid_distill/lowdata_summary_mean_std.csv)
- row for TS_ConvNeXtTiny_Residual
- QWK_mean = 0.9185819009963024
- QWK_std = 0.0009929178014660813

This is an aggregate statistic across three seeds, not a single-seed result. It is a model-level mean and standard deviation over the three saved seed runs for TS_ConvNeXtTiny_Residual, not a bootstrap estimate and not a validation metric.

The three saved seed results are:

- 42: 0.917438962528396
- 52: 0.9190749195807824
- 62: 0.9192318208797291

Mean = 0.9185819009963024

Rounded to four decimals, this is 0.9186.

The saved standard deviation is 0.0009929178014660813. Rounded to four decimals, this is 0.0010.

This exactly matches the manuscript value 0.9186 ± 0.0010 under the repository’s own aggregated summary definition.

For Accuracy = 85.47 ± 0.31, the source is also the same model-level summary table in [results/lowdata_hybrid_distill/lowdata_summary_mean_std.csv](results/lowdata_hybrid_distill/lowdata_summary_mean_std.csv). The exact value for TS_ConvNeXtTiny_Residual is Accuracy_mean = 0.854677565849228, which is 85.47% when multiplied by 100 and rounded to two decimals. The value is the mean across three seeds, not a single seed. The associated Accuracy_std is 0.0031463229928589786, which is 0.31% after multiplication by 100 and rounding to two decimals.

ROC-AUC and F1 are similarly model-level means from the same summary table and not from a single seed unless explicitly stated in the per-seed result file.

## 7) CHECKING THE 0.9186 VALUE

The three saved QWK values for TS_ConvNeXtTiny_Residual are:

- seed 42: 0.917438962528396
- seed 52: 0.9190749195807824
- seed 62: 0.9192318208797291

These produce the repository mean:

- mean = 0.9185819009963024

Rounded to four decimals: 0.9186

The repository standard deviation in the summary CSV is:

- QWK_std = 0.0009929178014660813

Rounded to four decimals: 0.0010

Therefore, the manuscript value 0.9186 ± 0.0010 is consistent with the saved seed results under the exact repository aggregation. No silent rounding error is present at the model-summary level; the match is exact after rounding to the manuscript precision.

## 8) FINAL STATUS

A. WHAT IS PROVEN
PROVEN

B. WHAT IS NOT PROVEN
NOT PROVEN

C. TEST-SET INDEPENDENCE STATUS
PARTIALLY PROVEN

D. SEED-SELECTION STATUS
PARTIALLY PROVEN

E. FINAL MODEL SELECTION STATUS
PROVEN

F. NUMERICAL DISCREPANCIES
NOT DETERMINED

G. BLOCKERS BEFORE FIRST NEW EXPERIMENT
PROVEN
