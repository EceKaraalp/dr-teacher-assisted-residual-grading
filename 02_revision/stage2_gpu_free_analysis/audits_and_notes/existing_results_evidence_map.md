# Evidence map – which number comes from which file

| Claim / metric | Value | Source | Verified how |
|---|---|---|---|
| Final model QWK mean±std | 0.9186 ± 0.0010 (0.91744/0.91907/0.91923) | `lowdata_final_summary_mean_std.csv`, `seed_*/test_metrics.json` | Recomputed from confusion matrix: max abs diff 0.0 (`no_gpu_analysis/outputs/per_seed_recompute_vs_recorded.csv`) |
| Final accuracy | 85.47 ± 0.31 % | same | same |
| Macro-F1 | 0.7034 ± 0.0094 | same | recomputed identical |
| ROC-AUC | 0.9590 ± 0.0011 | same | not recomputable (no probabilities) |
| Referable DR sens/spec | 0.964 ± 0.004 / 0.947 ± 0.003 | confusion matrices | `outputs/referable_dr_*.csv` |
| Cell-bootstrap 95% CI (seed 42) | QWK 0.893–0.941; acc 0.815–0.891 | confusion matrix | `outputs/cell_bootstrap_ci_per_seed.csv` (not paired, test-sampling only) |
| Params | 141,999,859 total / 27,824,454 trainable | `test_metrics.json` | contradicts manuscript "~149.8M" – unresolved |
| Latency | 6.70 ± 0.01 ms/img | `lowdata_final_summary_mean_std.csv` | hardware/batch not recorded in file |
| Gate2 vs EqualWeight | identical acc/QWK/F1; AUC 0.9583 vs 0.9453 | `teacher_student_inference_comparison.csv` | read directly; Gate2 test row equals seed-42 `test_metrics.json` |
| Teacher 1 | QWK 0.640 (manuscript) / 0.717 / 0.696 | three result sets (see inventory) | conflicting; cannot choose without provenance |
| Test used for selection | yes | `MODEL_SELECTION_FORENSIC_AUDIT.md` | sort by `QWK_mean` on test |
| Split | 2928/367/367 | `results/split_manifest.json` | recomputed from CSV with loader logic |
| DualExpert Exp2 / Exp5 | QWK 0.8189 / 0.7939, ECE 0.186 / 0.120 | `results/revision_existing_analysis_dual_expert_20261009/` | historical, not final model, label order ≠ manifest |

## Unresolved conflicts
1. Teacher 1/2 metrics differ across three result sets.
2. Manuscript param count (~149.8M) vs 141,999,859.
3. Whether Gate2/EqualWeight predictions are truly identical (aggregates only).
4. Fusion CNN-ViT QWK (0.923) above final model (0.9186).
