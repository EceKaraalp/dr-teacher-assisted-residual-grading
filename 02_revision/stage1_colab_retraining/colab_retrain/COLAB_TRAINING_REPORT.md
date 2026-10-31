# Colab Retraining Report: APTOS 2019 Teacher–Student Study

Run ID: `aptos_fullfactorial_20261011_v2`  
Code: `colab_retrain/` (`dr_common.py`, `dr_pipeline.py`, `run_all.py`, `gradcam.py`, `external_eval.py`, `Colab_Retrain_DR.ipynb`)  
Hardware: Google Colab, NVIDIA A100 (selected by the user)

> **Status of this document.** Sections 1–9 describe the *design and configuration*, read directly from the code. They are
> facts about what the run does. Section 10 (results) is a template: **no final results exist yet** because training
> was in progress when this report was written. The only observed numbers so far are the early log lines in
> Section 10.1. Nothing below should be quoted as a result until Section 10 is filled from `report.json` / `all_metrics.csv`.

---

## 1. Purpose

The previous manuscript and code disagreed on two points: (a) whether the final system uses a **learned gate** or an
**equal-weight** combination, and (b) the reported QWK values (0.902 vs 0.907). Reviewers additionally asked for
augmentation effects, the contribution of the weak teacher, speed/cost trade-off, Grad-CAM, and external validation.

This run retrains everything under one frozen protocol so these questions can be answered with measured numbers
(paired bootstrap, 3 seeds), instead of by argument. A single retraining does **not** by itself explain the old
0.902/0.907 discrepancy.

## 2. Data

| Item | Value |
|---|---|
| Dataset | APTOS 2019 (`train.csv` + `cache_224.npz`) |
| Image size | 224×224, RGB, PIL bilinear resize (cache verified against `train.csv`) |
| Classes | 5 DR grades (0–4) |
| Split | Stratified 80/10/10, `random_state=42` (two-step `train_test_split`) |
| Split sizes | train 2928 / val 367 / test 367 (**verified in Colab**) |
| Manifest check | train, val, test IDs all match the frozen original manifest (**verified: True ×3**) |
| Exact-duplicate groups | 123 found (resized-pixel duplicates; audit written to `duplicate_audit.csv`) |
| Class weights (train) | 0: 0.406, 1: 1.978, 2: 0.733, 3: 3.778, 4: 2.492 |
| Input scaling | float 0–255, CHW (same as the original loader) |
| Batch size | 16 (`BATCH` env var) |
| Augmentation | **None** for all teachers and for all student arms except `*_aug` |

Test data are never used for checkpoint selection or early stopping.

## 3. Models that are retrained

All models are trained from the pretrained/initial weights described below; none are loaded from the old paper results.

### 3.1 Baseline (Teacher 1), `Baseline_HybridCNNViT_Gate_Improved`
- Class `HybridCNNViTBase`; config `fusion_method=gate`, uncertainty refinement / ordinal head / prototype memory **off**.
- CNN branch: custom ResNet-18-style convolutional stack (64→128→256 channels) with multi-scale spatial-channel attention,
  followed by a 512-d projection. Trained **from scratch** (no ImageNet weights for Teacher 1; see `run_all.py` pretraining note).
- Trained once, seed 42, no augmentation.

### 3.2 Teacher 2, `Improved_Anchored_SpectralNorm` (three separately trained objectives)
- Components: frozen/anchored Teacher 1 + `AdvancedHybridModel` (config `Adv_Idea_5_SpectralNorm`) + fusion.
- `AdvancedHybridModel`: **ResNet-50** (ImageNet1K_V1) CNN branch (2048-d) + **ViT-B/16** (`timm`, pretrained, 768-d),
  concatenation fusion (2816-d), spectral-normalised linear classifier.
- Fusion with Teacher 1 is trained **separately for each objective**, so each fusion rule is optimised for itself and the
  comparison is fair:

| Teacher-2 objective (`fusion_mode`) | Final teacher output | Fusion parameters trained? |
|---|---|---|
| `gate` | `p=(1−g)·p_T1 + g·p_T2`, learned gate `g` | Yes (13→64→32→1 MLP, dropout 0.1) |
| `equal_prob` | `p=(p_T1+p_T2)/2` | No (gate frozen) |
| `equal_logits` | `z=(z_T1+z_T2)/2` | No (gate frozen) |

- Teacher 1 parameters are frozen inside the ensemble; Teacher 2 backbone and (for `gate`) the gate are trained.
- Gate input (13-d): `p_T1` (5), `p_T2` (5), entropy of each (2), absolute confidence gap (1).
- Checkpoints: `teachers/fusion_<mode>/Improved_Anchored_SpectralNorm/best_model.pth`.

### 3.3 Student, ConvNeXt-Tiny residual system (`ResidualTeacherStudentModel`)
- Student backbone: torchvision `convnext_tiny`, ImageNet1K_V1, 5-class head.
- The (frozen-checkpoint) teacher ensemble is loaded inside the system. Two student-side combination modes:

| `mix` | Output |
|---|---|
| `gated` (original proposed model) | `z = z_t + 0.70 · g · (z_s − mean(z_s))`, learned gate `g` (13→32→1) with conflict brake |
| `equal` | `z = (z_t + z_s)/2`, no student gate |

- Conflict brake: if teacher confidence > 0.82 and teacher/student argmax disagree, `g ≤ 0.66·(1 − teacher_conf)`.
- Student checkpoints store only student + student-gate weights; teacher weights are reloaded from the teacher checkpoint.

## 4. Experiment matrix (11 student arms × 3 seeds = 33 student runs)

Seeds: 42, 52, 62.

| # | Arm | Teacher | Student mix | Augmentation |
|---|---|---|---|---|
| 1 | `teacher_gate__student_gated` | gate-trained T2 | gated | no |
| 2 | `teacher_gate__student_equal` | gate-trained T2 | equal | no |
| 3 | `teacher_equalprob__student_gated` | equal-prob-trained T2 | gated | no |
| 4 | `teacher_equalprob__student_equal` | equal-prob-trained T2 | equal | no |
| 5 | `teacher_equallogit__student_gated` | equal-logits-trained T2 | gated | no |
| 6 | `teacher_equallogit__student_equal` | equal-logits-trained T2 | equal | no |
| 7 | `teacher_t1__student_gated` | Teacher 1 only (post-hoc) | gated | no |
| 8 | `teacher_t1__student_equal` | Teacher 1 only (post-hoc) | equal | no |
| 9 | `teacher_t2__student_gated` | Teacher 2 only (post-hoc) | gated | no |
| 10 | `teacher_t2__student_equal` | Teacher 2 only (post-hoc) | equal | no |
| 11 | `teacher_gate__student_gated_aug` | gate-trained T2 | gated | **yes** |

Note: arms 7–10 use the gate-trained T2 checkpoint post-hoc (`t1` restores the original Teacher 1 weights; `t2` uses T2 logits only).
They answer "does the weak/second teacher contribute?", not "what if it were trained alone?".

## 5. Hyper-parameters

### 5.1 Common to all trainers
| Item | Value |
|---|---|
| Optimiser | AdamW, weight decay 1e-4 |
| LR schedule | CosineAnnealingWarmRestarts, `T_0=10`, `T_mult=2` (stepped per epoch) |
| Gradient clipping | global norm 1.0 |
| Class weighting | inverse-frequency (Section 2) |
| Model selection / early stopping | **validation QWK** (best checkpoint saved, patience counter on QWK) |
| Resume | checkpoint every epoch; same `RUN_ID` resumes |

### 5.2 Per-model

| Model | Max epochs | LR | Patience | Loss |
|---|---|---|---|---|
| Baseline (T1) | 110 | 1e-3 | 15 | class-weighted cross-entropy |
| Teacher 2 (each of 3 objectives) | 120 | 1e-4 | 18 | see 5.3 |
| Student (each arm/seed) | 70 | 2e-4 | 10 | see 5.4 |

### 5.3 Teacher 2 loss
`L = main + λ·noninf + aux`, minus `β·H(g)` for `gate` only.
- `main`: class-weighted NLL on `log p_final` (CE on logits for `equal_logits`).
- `noninf = ReLU(main_i − base_i + margin)`; `base_i` = Teacher 1 loss per sample (non-inferiority to Teacher 1).
- `λ`: 1.5 → 0.6 (linear over epochs); `margin`: 0.02 → 0.005.
- `β` (gate entropy) = 0.02; gate entropy term only when `fusion_mode=gate`.
- `aux`: contrastive/prototypical auxiliary losses only if enabled in the config (not for `Adv_Idea_5_SpectralNorm`).

### 5.4 Student loss
`L = CE + λ_ni·noninf + 0.30·KD + 0.12·ordinal − 0.01·H(g)` (entropy term only for `gated`)
- `noninf = ReLU(CE_student_system − CE_teacher + margin)`; `λ_ni`: 0.8 → 0.25; `margin`: 0.010 → 0.003.
- KD: KL to teacher logits, temperatures [2, 4], weights [0.6, 0.4], scaled by `T²`.
- Ordinal: MSE between predicted and true class CDFs.

### 5.5 Student-arm augmentation (arm 11 only; training split only)
Horizontal flip p=0.5; vertical flip p=0.5; rotation uniform in [−20°, 20°] (bilinear, zero fill);
brightness/contrast factors uniform in [0.9, 1.1]. Validation and test are never augmented.

## 6. Training protocol decisions to be aware of
- **Teachers are trained once (seed 42).** Student variance (3 seeds) therefore excludes teacher-seed variance.
- **Original BatchNorm/dropout behaviour kept:** `FREEZE_TEACHER_MODE=False`, i.e. the frozen teacher stays in train mode
  during student/gate training, exactly as in the original code. A frozen-eval variant is not run.
- **Pretraining:** student ConvNeXt-Tiny and Teacher-2 ResNet-50/ViT-B/16 use ImageNet weights; Teacher 1 is trained from scratch.
  Loading failures raise errors rather than silently using random weights.
- **Historical caveat:** earlier model-family selection used test comparisons. This run does not erase that exposure;
  here test data are used only for the predeclared comparisons.

## 7. Evaluation

Stage `analyze` computes, on the frozen test set (and val):
- Metrics: QWK, accuracy, macro precision/recall/F1, weighted F1, macro specificity, weighted OvR ROC-AUC, MAE of grade,
  within-one-grade accuracy, multiclass Brier score, NLL, 15-bin ECE; per-class precision/recall/specificity/AUC/F1/confusion matrix.
- Statistics: paired, class-stratified percentile bootstrap, 2000 replicates, 95% CI.
- Pre-specified comparisons: gated vs equal student per teacher objective; teacher fusion rules against each other;
  T1-only / T2-only contribution; augmentation vs no augmentation; student-only vs full system.
- Duplicate sensitivity: secondary test QWK excluding test images with exact duplicates in train/val.
- Efficiency: forward-pass latency (batch 1 and 16) and parameter counts for student only, teacher ensemble, and full system
  (seed-42 gated arm). Preprocessing and I/O excluded.
- Grad-CAM (`gradcam.py`) and external evaluation (`external_eval.py`) are separate cells. External evaluation needs a
  user-supplied dataset; without one there is **no external validation**.

Outputs (Drive: `MyDrive/DR_Research/...`): `report.json`, `all_metrics.csv`, `student_summary_mean_sd.csv`,
`detailed_metrics.json`, `duplicate_audit.csv`, per-sample `predictions_*.csv`, `env.json`, `run_config.json` per run.

## 8. Mapping to reviewer comments

| Reviewer request | How the run addresses it | Limit |
|---|---|---|
| R2: no data augmentation | Paired augmentation ablations for both the gated sensitivity model and the manuscript-aligned equal-logit/equal-student model | Teachers are not retrained with augmentation |
| R2: weak teacher justified? | Arms 7–10 (T1-only, T2-only), teacher fusion arms 1–6 | Post-hoc, single teacher seed |
| R2: speed vs accuracy | Latency/params table (student vs full system) | Forward pass only, one GPU |
| R2: Grad-CAM | `gradcam.py` | Qualitative; not a proof of clinical relevance |
| R2/R1: other datasets | `external_eval.py` | Only if an external dataset is provided |
| Gate vs equal-weight | Arms 1–6, paired bootstrap | Does not explain old 0.902/0.907 gap by itself |

## 9. Compute and run-time notes
- GPU units: A100 ≈ 5.3 units/hour (observed in Colab UI). User had ≈200 units at the time of writing.
- Measured so far (A100): Teacher 1 ≈ 5 s/epoch; Teacher 2 ≈ 32 s/epoch. Student epoch time is **not yet measured**.
- Early stopping makes total time shorter than the epoch maxima; no total-time estimate is claimed.
- Sessions can be interrupted; progress mirrors to Drive about every 10 minutes and resumes with the same `RUN_ID`.

## 10. Results

Run `aptos_fullfactorial_20261011_v2`, NVIDIA A100-SXM4-40GB. All 33 student runs and the analysis stage completed. Test split n = 367. Source files: `analysis/student_summary_mean_sd.csv`, `analysis/report.json`.

### 10.1 Training log observations
| Model | Observation |
|---|---|
| Baseline (T1) | Early-stopped at epoch 25 (patience 15); best validation QWK **0.7262** (epoch 10); ~5 s/epoch |
| Teacher 2, gate | ~32 s/epoch; validation QWK 0.7770 → 0.8172 in epochs 1–2 |
| Students | ~24–26 s/epoch; early stopping at epochs 12–42 (patience 10) |

Teacher training log (selection on validation QWK; these are **validation** values, not test results):

| Teacher | Stopped at epoch | Best epoch | Best val QWK | Val acc at best | ~s/epoch |
|---|---|---|---|---|---|
| Baseline (T1) | 25 / 110 | 10 | 0.7262 | 0.6785 | 5 |
| Teacher 2, gate | 40 / 120 | 22 | 0.8770 | 0.8338 | 32 |
| Teacher 2, equal_prob | 24 / 120 | 6 | 0.8845 | 0.8392 | 32 |
| Teacher 2, equal_logits | 46 / 120 | 28 | 0.8929 | 0.8556 | 32 |

Notes: the baseline is clearly weaker than Teacher 2 (val QWK 0.73 vs 0.88–0.89) and its validation curve is noisy (e.g. 0.65 → 0.60 → 0.72 between epochs 11–15). Teacher 2 validation loss rises steadily after about epoch 9 while validation QWK stays flat, which indicates over-fitting of the probabilities; this is most pronounced for equal_logits (train loss ≈0.04, val loss ≈2.1–2.4). Checkpoint selection on QWK is therefore important. The best epoch of equal_prob (6) is early, so its result is more sensitive to a single epoch. The teachers use a single seed (42).

### 10.1b Teacher models, test set (single seed 42, n = 367)
| Model | Acc | Macro-F1 | QWK | ROC-AUC | ECE (15 bins) | NLL |
|---|---|---|---|---|---|---|
| T1 baseline | 0.6894 | 0.4757 | 0.7020 | 0.9048 | 0.0848 | 0.857 |
| T2 SpectralNorm (gate-trained) | 0.8229 | 0.6622 | 0.8805 | 0.9584 | 0.1526 | 1.346 |
| Equal-probability mixture T1+T2 (equal-prob trained) | 0.8474 | 0.6970 | 0.9059 | 0.9553 | 0.0908 | 0.556 |
| Equal-logit mixture T1+T2 (equal-logit trained) | 0.8392 | 0.6788 | 0.8855 | 0.9601 | 0.1299 | 0.894 |
| Learned probability gate T1+T2 | 0.8229 | 0.6622 | 0.8805 | 0.9509 | 0.1262 | 0.678 |

- The learned-gate teacher has the same accuracy, macro-F1 and QWK as T2 alone (0.8229 / 0.6622 / 0.8805), while NLL, ECE and AUC differ slightly. This matches the low gate values reported in 10.3: the gate effectively reproduces T2's decisions and adds little from T1.
- The equal-probability mixture is the best teacher on accuracy and QWK (0.9059) and has the best calibration (ECE 0.091, NLL 0.556). T1 alone is much weaker (QWK 0.702).
- Teacher QWK values are single-seed and have no confidence interval here. Differences of a few hundredths are not distinguishable from seed variation.

### 10.2 Student arms, test set (mean ± SD over seeds 42/52/62)
| Arm | Accuracy | Macro-F1 | QWK | ROC-AUC |
|---|---|---|---|---|
| teacher_equallogit / equal | 0.8474 ± 0.0109 | 0.6947 ± 0.0092 | 0.9014 ± 0.0078 | 0.9696 ± 0.0026 |
| teacher_equallogit / gated | 0.8429 ± 0.0031 | 0.6921 ± 0.0056 | 0.8890 ± 0.0027 | 0.9601 ± 0.0005 |
| teacher_equalprob / equal | 0.8211 ± 0.0063 | 0.6651 ± 0.0104 | 0.8875 ± 0.0158 | 0.9676 ± 0.0049 |
| teacher_equalprob / gated | 0.8383 ± 0.0031 | 0.6833 ± 0.0091 | 0.8921 ± 0.0170 | 0.9570 ± 0.0016 |
| teacher_gate / equal | 0.8510 ± 0.0110 | 0.7088 ± 0.0217 | 0.9010 ± 0.0100 | 0.9664 ± 0.0027 |
| teacher_gate / gated | 0.8292 ± 0.0031 | 0.6824 ± 0.0044 | 0.8897 ± 0.0024 | 0.9482 ± 0.0004 |
| teacher_gate / gated + augmentation | 0.8265 ± 0.0042 | 0.6752 ± 0.0073 | 0.8868 ± 0.0019 | 0.9475 ± 0.0014 |
| teacher_t1 / equal | 0.8329 ± 0.0126 | 0.6807 ± 0.0112 | 0.9013 ± 0.0091 | 0.9579 ± 0.0085 |
| teacher_t1 / gated | 0.7384 ± 0.0732 | 0.5542 ± 0.1325 | 0.7628 ± 0.1015 | 0.9204 ± 0.0270 |
| teacher_t2 / equal | 0.8456 ± 0.0134 | 0.6944 ± 0.0257 | 0.9045 ± 0.0087 | 0.9691 ± 0.0018 |
| teacher_t2 / gated | 0.8283 ± 0.0027 | 0.6776 ± 0.0026 | 0.8870 ± 0.0021 | 0.9565 ± 0.0002 |

### 10.3 Gated vs equal-weight
- Retrained comparison, same seed, same teacher objective (gate teacher): equal-weight students scored higher than gated students on QWK, accuracy and AUC. Seed 42 test: QWK 0.8922 (gated) vs 0.9047 (equal), delta −0.0125, 95% CI [−0.0319, 0.0064]; accuracy delta −0.0218, CI [−0.0490, 0.0054]. The intervals include zero.
- Post-hoc inference swap on the same gated checkpoint (seed 42, test): QWK 0.8922 (gated) vs 0.8939 (equal-weight); only a few predictions differ (5 on validation).
- Gate statistics (gate teacher, seed 42, validation): mean 0.090, median 0.020; 85.8% of images below 0.2, none above 0.7. The gate is concentrated at low values, so the learned weighting is far from adaptive. Which branch the low values favour was not verified.
- Conclusion supported by the data: no evidence that the learned gate improves over equal weighting. The earlier 0.902 vs 0.907 discrepancy is within seed-level variation (SD up to 0.017 QWK).

### 10.4 Teacher contribution (Reviewer 2)
- T1-only gated students are unstable (QWK 0.763 ± 0.101). T1-only equal students reach 0.901, similar to other equal arms.
- T2-only equal (0.9045) is at least as good as the combined teachers (0.9010 gate/equal; 0.9014 equal-logits/equal). No measured benefit from adding the weak teacher.
- Teacher pairwise test (single teacher seed): equal-logits vs equal-probability QWK delta −0.0204, CI [−0.0588, 0.0112]; gate-trained vs equal-probability-trained delta −0.0254, CI [−0.0529, −0.0011].

### 10.5 Augmentation (Reviewer 2)
The completed original augmentation comparison was on the gate-trained teacher/gated-student arm: seed-42 test QWK 0.8922 without vs 0.8847 with augmentation, delta −0.0074, CI [−0.0172, 0.0004]; three-seed means were 0.8897 vs 0.8868. This does **not** test augmentation on the manuscript's final equal-logit teacher/equal-student inference configuration.

A matched augmentation arm, `teacher_equallogit__student_equal_aug`, has now been added to the Colab package. It keeps the final model configuration, split, losses, optimizer, and seeds fixed relative to `teacher_equallogit__student_equal`, varying only the student-training augmentation policy. **Run this arm and its paired analysis before using augmentation results as evidence about the manuscript's final model.**

### 10.6 Latency (A100, batch 1)
| System | ms/image | images/s |
|---|---|---|
| Student only | 6.12 | 163.4 |
| Teacher ensemble | 18.45 | 54.2 |

The teacher ensemble is about 3.0× slower. Parameter counts and larger batch sizes are in `report.json` (`latency_and_params`).

### 10.7 Data audit
123 exact-duplicate groups; 26 test images have a duplicate in train/validation; 30 duplicate groups carry conflicting labels. Results were not recomputed on a de-duplicated test set in the headline tables; see `qwk_test_excluding_train_duplicates` in `report.json`.

### 10.8 Open items
- `system_vs_student_only` reports `qwk_student_only = 0.0` with 343 of 367 test predictions different (seed 42). This looks like a collapsed student-only output or a reporting artefact and is **unverified**. Do not cite it until checked.
- Grad-CAM figures: not reviewed in this report.
- External evaluation: not run. No external dataset was supplied, so there is no external validation.
- Student ECE and per-seed values are in `all_metrics.csv` and were not transcribed here.
- The three gated augmentation runs were retrained after their checkpoints were lost to a full Google Drive quota; with fixed seeds the re-run reproduced the earlier summary values to four decimals (QWK 0.8868 ± 0.0019). The manuscript-aligned equal-logit/equal-student augmentation arm was added afterward and remains to be run.

## 11. Known limitations
1. Single dataset (APTOS 2019) unless an external set is provided.
2. Single teacher seed (42); only students use 3 seeds.
3. Test-set exposure in earlier model-family selection (historical).
4. Exact duplicates exist in APTOS (123 groups); reported with a sensitivity analysis, not removed from the main split.
5. Smoke test passed (`SMOKE OK`) but used 600 images and 1 epoch; it validates the pipeline, not the results.
6. Training was not run on the local machine; local checks were limited to syntax/Problems and split verification.
