# Server checkpoint retrieval & verification plan (plan only — nothing fetched or run)

## Files needed
| # | File (server path to be confirmed) | Purpose | Must be matched to |
|---|---|---|---|
| 1 | Teacher 1 `best_model.pth` (`results/improved_unique_anchored_hybrid/Baseline_*/`) | teacher used in final distillation | experiment id of the 0.640-QWK run vs 0.717 retrain; seed; `split_manifest.json` |
| 2 | Teacher 2 `best_model.pth` (`.../Improved_Anchored_SpectralNorm/`) | frozen teacher (and resume candidate) | same; note the local partial (2-epoch) files are a different, unfinished run – do not mix |
| 3–5 | Student `best_model.pth` for `TS_ConvNeXtTiny_Residual`, seeds 42, 52, 62 | Grad-CAM, calibration, student-only evaluation | `results/lowdata_hybrid_distill/seed_*` test_metrics (QWK 0.91744/0.91907/0.91923) |
| 6 | Any Gate2 / EqualWeight inference artefacts, per-sample prediction/probability files | R1.1, paired tests | seed 42 comparison CSV |
| 7 | Training logs, config/hyper-parameters, notebook version actually run | provenance, loss definitions | — |
| 8 | `checkpoint_current.pth` for Teacher 2 (if training had continued on the server) | resume | — |

## After download (before any use)
1. Copy to a **new** read-only folder; never overwrite local checkpoints.
2. SHA256 for each file; record in a manifest with size, mtime, source path.
3. `torch.load(map_location="cpu", weights_only=True)`; list keys; check state-dict key set and shapes against the model definition; report epoch, best QWK, optimizer/scheduler presence.
4. Count parameters per component (student, each teacher, total) → resolve the 8.2M/113.8M vs 114.18M discrepancy.
5. CPU re-evaluation on the manifest test split; compare accuracy/QWK/confusion matrix with the recorded `test_metrics.json` (must match exactly or explain). Save per-sample labels/probabilities.
6. Only then: calibration, paired bootstrap, Grad-CAM, latency on one documented machine.
7. Teacher 2 resume only after items 1–5 and explicit approval; resume gaps known: no early-stop counter, no RNG state, `best_qwk` lags one epoch.

## Experiments still needing GPU/data
Teacher-2-only vs T1+T2 KD, controlled loss ablation, logit aggregation, normalization, augmentation, data-fraction (10/25/50/100 %), external dataset(s), full-system latency, validation-only model selection.
