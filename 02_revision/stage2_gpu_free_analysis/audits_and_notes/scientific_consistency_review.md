# Scientific consistency review

Evidence sources: `results/`, `MODEL_SELECTION_FORENSIC_AUDIT.md`, notebooks, `no_gpu_analysis/outputs/*.csv`, submitted Word, `0_main.tex`. No training/GPU was used.

## 1. Teacher 1 QWK = 0.640 / 0.717 / 0.696
| Value | Source | Acc | Split/seed/checkpoint info |
|---|---|---|---|
| 0.640 | `results/teacher_student_compare/` (and Table 1 / Word) | 0.632 | experiment id not stored with split manifest; seed not recorded; checkpoint not available |
| 0.717 | `results/teacher_student_compare_retrain/` | 0.695 | a retrain; seed/split not recorded; checkpoint not available |
| 0.696 | local reproduction 2026-10-09 (`D:\dr_env\repro_results`), manifest split, test | 0.540 | seed/split known, checkpoint exists locally (not the paper's) |

**Result:** the three runs cannot be matched to a common experiment id/seed/split/checkpoint; they are distinct trainings. **No single value is chosen.** Teacher 2 shows the same pattern (0.901 / 0.886 / unfinished).
**Text change (v2 does not yet apply, pending decision):** "Teacher 1 test QWK varied between 0.640 and 0.717 across three independent training runs (0.640 in the reported run); because the teacher is trained once and frozen, the paper's results are conditional on that particular teacher." The v1 Limitations sentence already notes teacher variability.

## 2. Parameter count (~149.8M vs 141,999,859)
- `test_metrics.json` (final model): trainable 27,824,454; `param_total` 141,999,859; frozen = 114,175,405 (the teachers).
- Table 5 teachers: ~8.2M + ~113.8M = 122.0M; 122.0 + 27.8 = **149.8M** → the 149.8M figure is the arithmetic sum of the table entries, not a counted number. The code count implies the frozen teacher set is 114.18M, i.e. **it does not reconcile** with 8.2M + 113.8M. Possible causes (unverified): Teacher 1 or Teacher 2 counts quoted for a different variant, or shared/untracked modules in the code count.
- Separation: **student-only 27.82M**; **full system total 142.0M (code)**, of which 27.8M trainable and 114.2M frozen.
- v2 Table 5: Full system trainable corrected to 27.8M (previously 142.0M, which contradicted the metrics file); note added that teacher rows are pending recount. Teacher checkpoints are required to resolve it.

## 3. Gate2 vs EqualWeight
`results/teacher_student_inference_comparison.csv`: accuracy, F1, QWK identical (0.9174, seed 42), AUC 0.9583 vs 0.9453. Identical aggregate metrics are compatible with identical labels but do **not prove** it; different AUC shows different probabilities (expected for different fusion). Gate2 test row equals the seed-42 `test_metrics.json` row, so "Gate2" is likely the mode used for the saved results – not confirmed. Per-sample predictions are unavailable → statement of "100 % label agreement" stays struck out. Text: "The two fusion modes produced identical aggregate accuracy, macro-F1 and QWK in the saved comparison (seed 42) but different AUC; sample-level agreement could not be verified."

## 4. Superiority vs CNN–ViT fusion
Fusion CNN–ViT standalone QWK 0.923 (single run) vs proposed 0.9186 ± 0.0010 (3 seeds). Unsupported claims: "most favourable QWK among evaluated configurations", "performing favorably relative to compared baselines", "consistent improvements over standalone backbones". v2 replaces them (Conclusion, Backbone paragraph). Suggested Abstract sentence (not edited in tex because the tex has no abstract section I verified; Word abstract needs the same): "The distilled student reaches a QWK comparable to the strongest standalone baseline (0.919 vs 0.923) with a 27.8 M-parameter student."
Remaining wording to review in v2: Discussion line "appears to yield consistent improvements over standalone backbones" (not changed – requires author's check of Table 1/3 numbers).

## 5. Test set used for model selection
Evidence: `MODEL_SELECTION_FORENSIC_AUDIT.md` (families ranked by `QWK_mean` from test results). v2: (a) Data Partitioning paragraph (already in v1), (b) **new "Model-Selection Protocol and Its Consequence" paragraph in the protocol section**, (c) Limitations (v1). Interpretation: all test numbers are optimistic; differences ≲ seed SD not interpreted. Still to do after training: select on validation only and/or evaluate on an untouched or external set.

## 6. Why calibration, paired tests, Grad-CAM are blocked
- Final student: no per-sample probabilities saved (only confusion matrices and aggregate metrics) → no ECE/Brier/NLL, no reliability diagram, no paired bootstrap/McNemar vs baselines.
- Final-model checkpoints are on the unreachable school server (local copy absent) → no inference, no Grad-CAM.
- No Grad-CAM file exists anywhere in the repo; the original Grad-CAM paragraph is struck out.
(An archived DualExpert model has ECE/NLL, but it is a different model and must not be cited as the final model.)

## 7. Re-check of earlier CPU analysis numbers (Task 5)
| Item | Value | File | Method | Notes |
|---|---|---|---|---|
| Acc, QWK, macro-F1 from confusion matrices, 3 seeds | match recorded values, diff 0 | `no_gpu_analysis/outputs/per_seed_recompute_vs_recorded.csv` | recompute from CM | validates CMs, not predictions |
| Referable DR (grade≥2) | sens 0.964 ± 0.004, spec 0.947 ± 0.003 | `referable_dr_*.csv` | collapse CM to 2×2; **± = SD across 3 seeds (n=3)** | not a CI |
| Per-class recall / F1 | Severe recall 0.298 (≈0.30), F1 0.399 (≈0.40); Mild 0.595, Moderate 0.883, Proliferative 0.689, No DR 0.978 | `per_class_*.csv` | CM; mean over seeds | |
| Bootstrap | seed-42 QWK 95 % CI 0.893–0.941 (≈0.89–0.94), 2000 reps, seed 20261010 | `cell_bootstrap_ci_per_seed.csv` | multinomial resampling of CM cells | single-model **uncertainty from test-set sampling**; **not paired**, no model comparison; ignores training variance |
Do not confuse: seed SD (training variability), bootstrap CI (test-sample variability), neither is a significance test.
