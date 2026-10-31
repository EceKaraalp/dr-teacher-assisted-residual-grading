REVIEWER-COMPLETE CONTROLLED RETRAINING PACKAGE

Purpose
  Produce new, separately identified controlled results for the reviewer
  questions. This is not a way to recreate an untouched APTOS test set.
  The original combined-output training objective is preserved, rather than
  silently moving distillation/ordinal/hinge losses onto the isolated student.
  No old experiment output or checkpoint is overwritten or deleted.

Colab setup
  Open DR_Reviewer_Complete_Colab.ipynb in Colab and select a GPU runtime.
  Create MyDrive/DR_Reviewer_Complete/ and upload:
    DR_Reviewer_Complete_Colab.zip   (code, frozen split, tests, protocol)
    DR_Reviewer_Complete_Data.zip    (verified train.csv and cache_224.npz)
    B. Disease Grading.zip          (official IDRiD archive; final evaluation)
  Use a new RUN_ID for this package. Do not reuse old experiment directories.
  Execute cells in order. The training cell runs all unfinished jobs by
  default; set JOBS_THIS_SESSION=3 to stop after three jobs and resume later.
  After disconnection: reconnect GPU, rerun setup, checks, and training.
  The notebook restores completed dependencies and the last persisted epoch.
  Keep the same code, dependencies and RUN_ID. A version/protocol mismatch
  stops explicitly instead of combining incompatible results.

Data
  3662 labeled APTOS images; fixed historical split 2928/367/367.
  Cache/CSV SHA-256 is checked; pixel cache alignment is not inferred solely
  from matching class labels. Original PNG files are not required in Colab
  for this pre-resized protocol. Retinal preprocessing is specifically LAB-L
  CLAHE at cached 224 resolution plus ImageNet normalization, not a full-size
  retinal preprocessing study.
  Training budgets are nested stratified subsets of the TRAIN partition,
  nominally 10%, 25%, 50%, 100%. Actual sample IDs/counts/class weights are
  saved. Both teachers and the student use only the corresponding subset.
  Validation and test partitions remain fixed. ImageNet pretraining uses
  external generic-image data; label-budget claims concern APTOS labels.

Frozen matrix: 99 training jobs
  Every setting uses seeds 42, 52, 62; no best seed is selected.
  Raw 100%:
    two teacher stages and 11 student arms per seed = 39 jobs.
    joint_ce              CE on the original joint gated output
    joint_single_kd       CE + predefined T=2 KD
    joint_multi_kd        CE + T=2/4 multi-temperature KD
    joint_ordinal         previous multi-KD + ordinal CDF
    joint_hinge           multi-KD + margin hinge, without ordinal
    joint_full_no_entropy multi-KD + ordinal + hinge, no gate entropy
    joint_full            original complete objective including gate entropy
    student_ce            independent ConvNeXt CE baseline, no teachers
    teacher2_only         same matched advanced teacher; no T1 decision signal
    augmentation          full objective + student-training augmentation
    teacher_eval          full objective with frozen teacher eval-mode buffers
  ImageNet-normalized 100% and retinal-processed 100%:
    retrain both teachers + joint_full + student_ce per seed = 24 jobs.
  Raw 10%, 25%, 50%:
    retrain both teachers + joint_full + student_ce per seed = 36 jobs.

Matched controls and interpretation
  Baseline teacher is the original random-initialized small hybrid; the
  advanced teacher and ConvNeXt use the original ImageNet pretraining.
  Teacher-2-only reuses the advanced weights from matched teacher training.
  It is not an independently trained T2 method. Its training implementation
  still executes the shared teacher container; it is NOT a pruning speed
  benchmark. Actual independent T1/T2 inference paths are benchmarked.
  The augmentation arm augments student-stage training inputs only; shared
  teachers are not retrained with augmentation. Report this scope explicitly.
  Geometric/photometric transforms are fixed in plan.py and applied only to
  training inputs. No validation/test augmentation is used.
  The teacher_eval control changes dropout as well as BN/SN buffer behavior;
  do not interpret it as a spectral-normalization-only ablation.
  No causal calibration benefit from SN alone can be inferred: there is no
  otherwise-identical teacher-training run without SN. Such a claim must
  remain removed.
  All joint-objective arms have identical student/gate initialization for
  a given seed. A single saved teacher pair is reused across matched arms.
  Full independent teacher/student seed crossing is not performed.
  Validation QWK selects the primary-training-output epoch. Student-only and
  equal-inference results are from that same checkpoint, not separately
  chosen epochs. This tests the original method, not a reoptimized student.

Persistent output and recovery
  Per run: config.json, history.json, best.pth, last.pth, complete.json.
  last.pth contains optimizer, scheduler, RNG, early-stopping state.
  Every completed epoch is copied to Drive atomically and hash checked.
  Student checkpoints omit shared teacher parameters but preserve their
  updated buffers. They therefore REQUIRE their matched teacher2 best.pth,
  which in turn requires teacher1. Never keep only student best.pth.
  Complete checkpoints are hash checked before reuse. Resume was locally
  tested against uninterrupted training, including dropout and shuffle RNG.
  Interrupted mid-epoch work is rerun from the previous persisted epoch.
  No runtime, GPU time, quota or reviewer-acceptance guarantee is possible.

Storage and runtime
  Plan for at least 85 GiB free in BOTH Colab runtime disk and Drive before
  beginning. Estimated retained checkpoint size is approximately 60-75 GiB.
  Drive FUSE disk_usage may not reflect account quota: check Drive manually.
  A normal 15 GB Drive account is insufficient for this complete matrix.
  Hash-verified epoch backups add I/O overhead. Do not delete old weights
  blindly to make space; back them up and verify separately first.
  One notebook orchestrates the entire matrix; one uninterrupted free-Colab
  session is not assumed. Measure the first completed jobs before estimating
  total GPU hours. CPU architecture checks are not evidence of GPU runtime.

Post-training cells
  Evaluate: all 99 jobs must be complete before any APTOS test evaluation.
    Writes aligned validation/test logits and per-path metrics.
  Analyze:
    Accuracy, macro-F1, QWK, weighted OVR AUC, NLL, multiclass Brier, ECE15,
    per-class counts/recall/precision, confusion and reliability figures.
    Fits separate positive scalar temperatures on validation NLL ONLY.
    Raw-logit, probability-average and validation-temperature fusion are
    reported without choosing a preferred method using test scores.
    Reports seed means/sample SD and paired image-bootstrap QWK differences
    (10,000 samples; shared IDs across seed pairs).
    Bootstrap is conditional on the trained checkpoints, is not patient
    clustered, and does not account for training or multiple-test uncertainty.
  Benchmark:
    Student, T1, T2, learned teacher ensemble, full equal, full gated.
    All three seeds, FP32, batches 1/16, 20 warm-ups, 100 repetitions.
    Only the measured path is resident on GPU; CUDA synchronization and
    allocated-memory peaks are recorded. Forward-only: no I/O/preprocessing.
  External:
    All 103 official IDRiD test images; 413 training images unused.
    New raw joint_full and independent-CE checkpoints, all three seeds.
    No target normalization/threshold fitting/TTA/calibration or seed choice.
    Grad-CAM: first test image per true grade in each dataset; seed 42 fixed.
    CNN branches only; forward equivalence checked; zero maps labeled.
    This is descriptive attribution, not lesion localization validation.

Remaining nonexperimental issues
  Retraining cannot supply missing patient/demographic/device metadata,
  prospective clinical evidence, or a newly independent geographic cohort.
  APTOS and IDRiD test results have been examined previously. This package
  does not restore fresh-holdout status or justify clinical deployment.
  Reference verification, AI-use disclosure, the incorrect Comment-12 reply
  and final Word formatting require separate manuscript/response edits.
  New numeric results must be inserted consistently only after these jobs
  actually finish; old archive numbers cannot be attributed to new models.

Architecture provenance
  Architecture/helpers were copied from the existing repository package.
  AdvancedHybridModel gains only a pretrained=False construction option for
  offline checks and complete-state loading; training defaults remain True.
  Fresh training never silently falls back to random pretrained backbones.
  Python source hashes, data hashes, configuration and installed dependency
  versions are recorded. PyTorch/torchvision are the matched Colab runtime
  pair and are intentionally not replaced by an unrelated wheel.
