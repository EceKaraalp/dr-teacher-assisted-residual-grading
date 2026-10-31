# Reviewer response matrix (R1: 21, R2: 5)

Categories: A = existing files suffice; B = CPU analysis of existing records; C = new training/GPU; D = external data; E = not verifiable → limitation/claim correction.
Response status: **Ready** / **Partial** (text/analysis done, experiment pending) / **Waiting**.
Item wording follows `docs/revision/REVISION_PLAN_3_DAY.md` (reviewer PDF itself not re-read today).

| # | Topic | Cat. | Evidence today | Done today (GPU-free) | Missing / blocker | Status |
|---|---|---|---|---|---|---|
| R1.1 | Gate2 vs EqualWeight "identical predictions" but different QWK | E | Available comparison is tied to an intermediate training/evaluation record and has not been reconciled with the final reported model | No intermediate aggregate value is used to revise the final-model result; the earlier identity/gate-collapse interpretation is not presented as independently verified | Final-model re-inference with archived per-sample predictions; if no change is found, retain the existing reported result | Waiting |
| R1.2 | Test used for model selection | E | Forensic audit | Limitation added to draft (test used to rank families) | Independent data (R2.1) is the only fix | Partial |
| R1.3 | Student-only vs full system | B/C | Saved metrics/code: 0.9186 QWK and 6.70 ms/image are full-system; 27.8M trainable, 142.0M total | Manuscript now identifies the full-system scope and states student-only metrics/latency/memory are unavailable | Separate student-only evaluation and standardized component-wise benchmark | Partial |
| R1.4 | Controlled loss ablation | C | Only historical non-controlled variants | – | Retrain with single-component changes | Waiting |
| R1.5 | Superiority vs standalone models (Fusion CNN-ViT 0.923 > 0.9186) | A/E | Single-run standalone QWK 0.923 vs full-system three-seed mean 0.9186 | Abstract, Results, Discussion, and Conclusion now state superiority is not established | Final proofreading; no paired multi-seed comparison | Partial |
| R1.6 | Usefulness of Teacher 1 | C | T1 QWK 0.640/0.717/0.696 across runs | Run-to-run variability recorded in draft | Teacher-2-only vs 2-teacher KD, same seeds | Waiting |
| R1.7 | Logit averaging rationale | C | Training code uses a confidence-gated probability mixture for teachers and a gated residual student output, not raw-logit averaging | Manuscript now documents the implemented inference rule and avoids claims about raw-logit averaging | Comparative fusion/calibration experiments were not performed | Partial |
| R1.8 | Calibration | B/C | Final model: no probabilities | Archived DualExpert ECE/NLL computed (not final model) | Final-model probabilities (needs inference) | Waiting |
| R1.9 | ImageNet normalization | C | Code: raw tensors, no normalization | – | Normalization ablation | Waiting |
| R1.10 | Low-data claim | C | All reported runs use the full training partition | Abstract, conclusion, and limitations now state that a low-data advantage was not evaluated | 10/25/50/100% training-fraction experiments | Partial |
| R1.11 | Non-inferiority loss definition | A | Notebook hinge compares full-system CE with detached teacher CE using a positive margin | Equation checked; term renamed teacher-relative superiority-margin hinge and gradient behavior clarified | Controlled ablation remains training-dependent | Partial |
| R1.12 | Reproducibility | A/E | Manifest, seeds, hyperparameters | Manifest created; checkpoints still absent | Release code+manifest; checkpoints on server | Partial |
| R1.13 | Params/latency | B | `param_total` 141,999,859; trainable 27,824,454; saved 6.70 ms timing is full-system test-loader timing | Table corrected to 4,851,623 + 109,320,773 + 3,009 frozen teacher params and 27,824,454 trainable params; latency scope disclosed | Standardized student-only/teacher-only and memory measurements | Partial |
| R1.14 | Statistical uncertainty | B/C | confusion matrices | Cell-bootstrap 95% CI per seed (QWK ≈0.89–0.94) | Paired bootstrap/tests need per-sample preds | Partial |
| R1.15 | Class-wise / referable DR | B | confusion matrices | Per-class + referable sens 0.964, spec 0.947 added to draft | Student vs full-system comparison | Partial |
| R1.16 | Fig. 2 inference mode | C | – | – | Regenerate for both modes | Waiting |
| R1.17 | Literature comparison protocol | E | Draft already says contextual | – | Same-protocol baselines or explicit caveat | Partial |
| R1.18 | Bibliography accuracy | A | Prior Crossref audit identified metadata/DOI errors | Applied verified corrections to `0_main.tex` for cao2020, liu2020, oh2022, minarno2022, nabavi2024, yang2025, and elassiouti2024 | Claim-level support and unaudited entries | Partial |
| R1.19 | Clinical/deployment claims | A | External/prospective validation, calibration, and device/population testing absent | Removed unsupported readiness, real-time, on-device, and calibration claims; expanded limitations | Final read-through | Partial |
| R1.20 | AI-use statement | A | – | **Not done** | Author-supplied statement | Waiting |
| R1.21 | Presentation/format | A | Efficiency table and result scopes checked | Table/figure scope captions and single-run vs multi-seed caveats updated | LaTeX compile and final proofreading | Partial |
| R2.1 | External dataset | D | None obtained | – | Obtain Messidor-2/IDRiD/DDR/EyePACS + checkpoints | Waiting |
| R2.2 | Augmentation | C | Paper: none used | – | Augmentation ablation | Waiting |
| R2.3 | Weak Teacher 1 | C | Variability across runs | Recorded | T2-only vs T1+T2 | Waiting |
| R2.4 | Speed vs accuracy | B/C | Saved 6.70 ms/image is full-system on RTX A5000, batch 16; timing includes loader/transfers/first batch | Corrected scope and stated that the result is not a standardized benchmark | Standardized student/full-system timing and paired accuracy comparison | Partial |
| R2.5 | Grad-CAM | C | No files | – | Needs student checkpoint | Waiting |

## Tally
- Fully solved GPU-free: **0**.
- Partially advanced GPU-free: **R1.1, 1.2, 1.3, 1.5, 1.6*, 1.8, 1.11, 1.12, 1.13, 1.14, 1.15, 1.17, 1.18, 1.19, 1.21, R2.4** (16; *R1.6/R2.3 only as variability note).
- Need new training/checkpoint analysis: R1.3 (student-only branch), 1.4, 1.6, 1.8, 1.9, 1.10, 1.16, R2.2, 2.3, 2.4, 2.5.
- Need external data: R2.1.
- Author-supplied statement and final QA still open: R1.20; R1.18 claim-level citation support; R1.21 compile/proofread.
