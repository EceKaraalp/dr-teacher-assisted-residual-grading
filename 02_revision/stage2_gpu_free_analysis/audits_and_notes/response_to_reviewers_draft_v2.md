# Authors' Response to Reviewers — DRAFT v2 (not for submission)

Layout follows the example response letter (General Comments → reviewer comment → bold response → revised text). Reviewer comments are paraphrased from `docs/revision/REVISION_PLAN_3_DAY.md` (the reviewer PDF could not be parsed); **replace with verbatim text before submission**. `[PENDING]` = experiment/evidence not yet available; `[LIMITATION]` = stated openly.

## General Comments
We thank the reviewers. Several requested experiments (external dataset, augmentation, controlled ablations, calibration, Grad-CAM, full-system evaluation) require model checkpoints and GPU time that are in progress; these are marked `[PENDING]` and will be completed and reported before resubmission. Only analyses that could be executed on the existing records are reported now. All changes are highlighted (cyan) in the revised manuscript.

## Reviewer 1

**R1.1 – Gate2 vs EqualWeight identical predictions but different metrics.**
**Response:** This comment depends on reconciling an intermediate Gate2/EqualWeight comparison with the final reported model and its archived outputs. Because the currently available comparison is tied to earlier training/evaluation records, we do not use its aggregate values to revise the final-model results at this stage. The question of whether the two configurations produced identical labels, and whether any probability-level difference explains a metric difference, is deferred until the final model outputs are re-evaluated. If the final re-evaluation shows no change, the existing reported result will be retained rather than replaced by an intermediate result. `[PENDING]` final-model re-inference with archived per-sample predictions.
*Revised text:* No new numerical result is added for this comment at this stage. The manuscript only avoids presenting the earlier identity/gate-collapse interpretation as independently verified; the comparison will be revisited after final-model evaluation.

**R1.2 – Test set used for model selection.**
**Response:** Confirmed. Epoch selection used validation QWK, but candidate model families were ranked by test QWK. The claim that the test set was untouched was removed; a "Model-Selection Protocol and Its Consequence" paragraph now follows the Data Partitioning paragraph, and the Limitations section repeats it. Test metrics are described as optimistic. `[PENDING]` validation-only selection and untouched/external evaluation.

**R1.3 – Student-only vs full system.**
**Response:** We rechecked the saved metrics and evaluation code. The reported three-seed QWK/accuracy and 6.70 ms/image timing are for the complete teacher--student system; the timed forward pass includes both teachers, the student, data loading, device transfers, and the first batch (batch size 16, NVIDIA RTX A5000). The ConvNeXt Tiny student component has 27,823,973 parameters, but student-only test metrics, latency, and memory were not separately recorded. The manuscript now labels the results as full-system and does not infer student-only deployment performance. `[PENDING]` separate student-only evaluation and standardized component-wise timing.

**R1.4 – Controlled loss ablation.** `[PENDING]` single-component-change retraining.

**R1.5 – Superiority claims.**
**Response:** The custom standalone Fusion CNN--ViT reports QWK 0.923 in a single run, compared with the full system's three-seed mean QWK 0.9186 ± 0.0010. The manuscript now states that these differently replicated results do not establish superiority and removes the general claim of consistent improvement over standalone backbones. No paired test was performed.

**R1.6 – Usefulness of Teacher 1.**
**Response:** Teacher 1 QWK differs across our runs (0.640, 0.717, 0.696) and cannot be matched to one experiment; this is stated as a limitation. `[PENDING]` Teacher-2-only vs two-teacher distillation, same seeds.

**R1.7 – Logit averaging rationale.**
**Response:** We checked the training/inference implementation and corrected the manuscript: teacher outputs are combined by a confidence-conditioned probability gate, and student logits enter through a learned residual gate. Direct equal-weight averaging of raw teacher logits is not the reported inference method. `[PENDING]` controlled comparisons of alternative teacher-fusion/calibration strategies; the manuscript makes no claim that these alternatives were tested.

**R1.8 – Calibration.**
**Response:** Calibration was not evaluated for the final model. Claims that spectral normalization or distillation improved calibration/probability estimates were removed or qualified in the manuscript. `[PENDING]` ECE, Brier score, NLL, and reliability diagram (requires final-model probabilities).

**R1.9 – ImageNet normalization.** `[PENDING]` normalization ablation (current pipeline uses raw tensors; stated in the paper).

**R1.10 – Low-data claim.**
**Response:** All reported runs use the full available training partition. The Abstract, Introduction, Conclusion, and Limitations now state that no reduced-data experiment was performed and that a low-data advantage is not established. `[PENDING]` controlled 10/25/50/100 % training-fraction experiments.

**R1.11 – Non-inferiority loss definition.**
**Response:** The equation was checked against the training notebook. The teacher loss is computed under `no_grad`, while the hinge uses full-system cross-entropy and a positive margin: `max(0, L_full - L_teacher + margin)`. The manuscript renames this a teacher-relative superiority-margin hinge rather than conventional non-inferiority and clarifies the optimized outputs. `[PENDING]` controlled ablation of this term.

**R1.12 – Reproducibility.**
**Response:** A frozen split manifest (2,928/367/367, seed-42 stratified) exists and a Data availability statement was added; code/manifest/checkpoint release `[PENDING authors' decision]`.

**R1.13 – Parameters and latency.**
**Response:** The model definitions and saved metrics give 27,824,454 trainable parameters and 114,175,405 frozen teacher parameters, for 141,999,859 total. Component counts are 4,851,623 (Teacher 1), 109,320,773 (Teacher 2), 3,009 (teacher gate), 27,823,973 (ConvNeXt Tiny), and 481 (student gate); the earlier 149.8 M resulted from incorrect teacher component counts. The manuscript corrects these counts and identifies 6.70 ± 0.01 ms/image as the full-system test-loader timing, not student-only latency. It records the RTX A5000, batch size 16, and timing overhead; this was not a standardized component-wise benchmark. `[PENDING]` student-only/teacher-only timing, memory, and a standardized same-hardware benchmark.

**R1.14 – Statistical uncertainty.**
**Response:** Added seed mean ± SD (training variability) and a confusion-matrix-resampling 95 % interval for seed 42 (QWK 0.893–0.941; 2,000 replicates), which reflects test-sample variability for one model and is not a paired comparison. `[PENDING]` paired bootstrap/McNemar (needs per-sample predictions).

**R1.15 – Class-wise / referable DR.**
**Response:** Added (v1 paragraph): referable-DR (grade ≥ 2) sensitivity 0.964 ± 0.004, specificity 0.947 ± 0.003 (3 seeds, SD); per-class recall No DR 0.978, Mild 0.595, Moderate 0.883, Severe 0.298 (F1 0.399), Proliferative 0.689. Severe-class performance is explicitly flagged as weak.

**R1.16 – Fig. 2 inference mode.** `[PENDING]` regenerate for student-only and full system.

**R1.17 – Literature comparison protocol.**
**Response:** Text already states the comparison is contextual because splits/augmentation/kappa definitions differ; values in Table 4 are copied from the cited papers and `[PENDING]` re-verification.

**R1.18 – Bibliography accuracy.**
**Response:** The previously recorded Crossref audit corrections were applied to the active manuscript: `cao2020` and `liu2020` DOIs; `oh2022` authors; `minarno2022` authors/pages; `nabavi2024` publication year/volume and author form; and the published versions/DOIs for `yang2025` and `elassiouti2024`. Claim-level support and entries not covered by the prior audit have not been verified `[PENDING]`.

**R1.19 – Clinical claims.** Claims of clinical utility, real-time/on-device use, and regulated-workflow readiness were removed or identified as unvalidated; the Limitations section now states the single-dataset, no-prospective/no-external-validation constraints. Final read-through `[PENDING]`.

**R1.20 – AI-use statement.** `[PENDING authors]` Confirm all generative-AI use, including language editing and this revision, before inserting an accurate journal-compliant statement; do not infer or invent author usage.

**R1.21 – Presentation/format.** The efficiency table now sits in its computational-efficiency subsection; table/figure captions distinguish full-system from student-only results, and mean ± SD/single-run comparisons are qualified. LaTeX compile and final typography/proofreading `[PENDING]`.

## Reviewer 2

**R2.1 – Other datasets.** `[LIMITATION now; PENDING]` external validation (e.g. Messidor-2/IDRiD/DDR/EyePACS) requires data access and checkpoints; Limitations/Future Work already state single-dataset evaluation.

**R2.2 – No data augmentation.** The zero-augmentation protocol is stated as a deliberate design choice, and its effect on results is not yet known. `[PENDING]` augmentation ablation.

**R2.3 – Weak Teacher 1 (≈63 %).** Teacher 1's standalone accuracy is low and variable across runs; the paper states it is not an independently strong classifier and its contribution is marginal (ensemble QWK 0.902 vs 0.901 for Teacher 2). `[PENDING]` Teacher-2-only vs T1+T2 to determine whether it contributes.

**R2.4 – Speed vs accuracy trade-off.** Student branch 6.70 ms/img; full-system latency not measured. `[PENDING]` same-hardware measurement and discussion.

**R2.5 – Explainability (Grad-CAM).** The earlier Grad-CAM paragraph was removed because no such analysis existed. `[PENDING]` Grad-CAM for the student (needs checkpoint).
