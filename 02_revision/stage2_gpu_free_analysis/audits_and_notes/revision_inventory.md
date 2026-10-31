# Revision inventory (GPU-free, 2026-10-10)

Scope: files that were physically inspected. Nothing was trained; no GPU/torch was used.

## 1. Frozen split
- `results/split_manifest.json`: train 2928, val 367, test 367 (stratified, `random_state=42`, two-stage split identical to `dataset_loader.py`). No ID overlap between splits.
- Not done: content-hash duplicate scan (would require reading all 3662 images).
- Manuscript was corrected to 2,928/367/367 (was 2,929/366/367). Original Word submission not re-compared (see change_log, "not done").

## 2. Final model (TS_ConvNeXtTiny_Residual)
| Item | Status | Path |
|---|---|---|
| Per-seed `test_metrics.json` (acc, QWK, F1, AUC, **confusion matrix**, params, ms/image) | present, seeds 42/52/62 | `results/lowdata_hybrid_distill/TS_ConvNeXtTiny_Residual/seed_*/` |
| Training history | present | same folders |
| Seed CSV / mean-std summaries | present | `results/lowdata_hybrid_distill/*.csv` |
| Per-sample predictions / probabilities / logits | **absent** | – |
| Checkpoints (student, teachers) | **absent locally** (not in git, server unreachable) | – |
| Gate2 vs EqualWeight comparison | present (aggregate only) | `results/teacher_student_inference_comparison.csv` |
| Model-selection evidence (test-QWK ranking) | present | `MODEL_SELECTION_FORENSIC_AUDIT.md` |

## 3. Teachers (three independent result sets – they disagree)
| Source | T1 test acc / QWK | T2 test acc / QWK |
|---|---|---|
| `results/teacher_student_compare/` (matches manuscript Table) | 0.632 / 0.640 | 0.831 / 0.901 |
| `results/teacher_student_compare_retrain/` | 0.695 / 0.717 | 0.812 / 0.886 |
| Local reproduction 2026-10-09 (D:\dr_env\repro_results) | 0.540 / 0.696 | not finished (2 epochs, val QWK 0.832) |

`results/teacher_student_compare/` also contains a `Student_ConvNeXt_Tiny` row with test acc 0.0518, QWK 0.0 (collapsed model, not the KD student).

## 4. Archived DualExpert predictions (`experiments_dual/Exp2,4,5,7`)
- Contain y_true/y_proba/confusion/training_log, **no sample IDs**; label order differs from the frozen manifest. Historical experiments, not the final model.
- Re-analysed on CPU: Exp2, Exp5 in `results/revision_existing_analysis_dual_expert_20261009/`. Exp4/Exp7 not completed (script stopped earlier).

## 5. Explainability
- No Grad-CAM image/file exists in the repo (searched `*gradcam*`). Manuscript Grad-CAM text is struck out.

## 6. Reproduction state (outside repo)
- `D:\dr_env\repro_results\improved_unique_anchored_hybrid`: Teacher 1 finished; Teacher 2 paused after 2 epochs (`checkpoint_current.pth` epoch idx 1; wait-counter/RNG not saved).

## 7. Reference documents available
`C:\Users\user\Desktop\Round1\`: submitted docx, reviewer PDF, `20250706_Response2reviewers.docx`, `BSPC-D-25-04358_manuscript_marked.pdf`, `proper sentences.docx`. The example response/marked manuscript were **not** read in this session; only the response format needs to be mirrored (to do before final submission).
