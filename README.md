# Teacher-assisted ordinal learning for diabetic retinopathy grading

Code, experiment notebooks, logged outputs and audit notes for the manuscript
*"Teacher-assisted ordinal learning for diabetic retinopathy grading: a controlled evaluation of gated residual and equal-weight inference"*.

This repository is provided so that reviewers and readers can inspect how the reported numbers were produced.
It separates the **original study** from everything done **during the revision**.

## Repository map

| Path | Contents |
|---|---|
| [`01_original_study/`](01_original_study) | Code, notebooks, results and notes from the original (pre-revision) study. |
| [`02_revision/`](02_revision) | Everything produced during the revision, organised by stage (below). |
| [`data/`](data) | `split_manifest.json`: the APTOS 2019 split definition. No images are redistributed. |
| [`CHECKPOINTS.md`](CHECKPOINTS.md) | Where model weights are kept and how to restore them. |

### `01_original_study/`
- `code/` – models (`models/`), losses (`losses/`), utilities, training scripts and the exported ConvNeXt-Tiny teacher–student packages (`export/`).
- `notebooks/` – training and comparison notebooks used in the original study.
- `results/` – logged outputs: `results_dir/` (the former `results/`), `experiments/`, `experiments_dual/`, `experiments_models/`.
- `docs/` – internal guides and summaries (several are in Turkish).

### `02_revision/`
| Stage | Folder | What it is |
|---|---|---|
| 0 | `stage0_historical_pipeline_reconstruction/` | Code reconstructing the executed pipeline from archived checkpoints, optimizer states and notebooks. |
| 1 | `stage1_colab_retraining/` | Early Colab retraining packages on the APTOS split (code, reference tables). |
| 2 | `stage2_gpu_free_analysis/` | Analyses that need no GPU: QWK discrepancy audit, Equal-vs-gated recomputation, archived-result verification, IDRiD external evaluation and Grad-CAM, prediction-ID repair. Includes `audits_and_notes/`. |
| 3 | `stage3_round1_99job_matrix/` | Code and notebook for the controlled 99-job matrix (3 seeds); `results/reviewer_complete_v1/` holds its calibration, matched A100 timing and Grad-CAM outputs. |
| 4 | `stage4_round2_30job_matrix/` | Code and notebook for the additional 30-job control matrix (independent Teacher 2, no spectral normalization, augmentation, Fusion CNN-ViT baseline); `results/colab_reviewer_round2_results/` holds logits, metrics, `analysis2/` summaries and `external2/` IDRiD outputs. |
| – | `docs/`, `figures/` | Revision plan, baseline/critical audits, evidence JSONs, and the final Figure 1 and Figure 2 images. |

## Protocol summary
- Dataset: APTOS 2019 with the historical split of 2,928 training / 367 validation / 367 test images; seeds 42, 52, 62.
- External evaluation: 103 IDRiD official test images, no adaptation.
- Checkpoint selection: validation QWK.
- Primary reported inference pathway: **Equal** (`z_Equal = (z_T + z_S) / 2`); gated residual inference is secondary. The choice was made post hoc and is not a confirmatory comparison.

## Reproduction notes
- Datasets are **not** included; obtain APTOS 2019 (Kaggle) and IDRiD from their original sources and follow their licences. `data/split_manifest.json` defines the split.
- Colab notebooks expect a Google Drive layout (`/content/drive/MyDrive/...`) and clone this repository.
- Python dependencies: `01_original_study/code/requirements.txt` (original) and the `requirements.txt` in each `stage*/code/colab_*` folder.
- Large binary outputs (model weights) are not in the git history; see [`CHECKPOINTS.md`](CHECKPOINTS.md).

## What is *not* here
- Model weights (`.pth`) – see `CHECKPOINTS.md`.
- Per-job logits of the round-1 99-job matrix; they remain on the Colab/Drive run directory (`reviewer_complete_v1`). Aggregated outputs are in `stage3_round1_99job_matrix/results/` and `stage4_round2_30job_matrix/results/.../analysis2/`.
- The APTOS/IDRiD images and the manuscript files.

Paths inside some logs and provenance files refer to the authors' local machine or Google Drive and are kept as originally recorded.
