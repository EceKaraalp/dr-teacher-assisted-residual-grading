# Change log (2026-10-10)

| File | Change | Evidence |
|---|---|---|
| `revised_manuscript_draft.tex` (copy of `0_main.tex`; original untouched) | Added `\hlcyan` paragraph "Class-wise and referable-DR view" after the confusion-matrix discussion | `no_gpu_analysis/outputs/*.csv` |
| same | Extended Limitations with test-set-selection disclosure and teacher run-to-run variability | `MODEL_SELECTION_FORENSIC_AUDIT.md`; three teacher result sets |
| `no_gpu_analysis/recompute_from_confusion_matrices.py` + `outputs/` | New CPU script; recomputed acc/QWK/F1 equal recorded values (diff 0.0) | run output |
| `revision_inventory.md`, `reviewer_response_matrix.md`, `existing_results_evidence_map.md`, `revision_priority_plan.md`, `response_to_reviewers_draft.md` | New | – |

## Not done (explicitly)
- Submitted Word vs LaTeX comparison; reading the example response docx/marked PDF.
- Bibliography verification, AI-use statement, formatting QA.
- Image-hash duplicate scan; Exp4/Exp7 archived analysis.
- Parameter-count/Table 5 correction in the manuscript (conflict only reported).
- No training, no GPU, no existing file overwritten.
