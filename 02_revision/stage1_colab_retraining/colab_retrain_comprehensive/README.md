# colab_retrain_comprehensive

Comprehensive retraining pack (APTOS 2019, ConvNeXt-Tiny residual student, T1 + T2 teachers). Notebook: `Colab_Retrain_DR_Comprehensive.ipynb`.

Upload `colab_retrain_comprehensive.zip` (forward-slash entries) or the unpacked folder to `MyDrive/DR_Research/`. Data: `MyDrive/DR_Research/data/{train.csv,cache_224.npz}`.

## Experiments (all resumable, mirrored to Drive)
| Exp | Setting | Purpose |
|---|---|---|
| E0_main | split 2929/366/367 (`SPLIT_MOVE=c0p0`), 12 arms x seeds 42/52/62 | gate vs equal, T1/T2 roles, augmentation on gated AND final model, latency, ECE, paired bootstrap. Reuses the earlier run `results_retrain/aptos_manuscript_final_split366_20261012_v1` |
| E_teacherseed52/62 | teacher seed 52/62, final arm | teacher-seed variance / reproduction of 0.9186 |
| E_split_c1p0..c4p0 | different validation image moved to train | split sensitivity (historical IDs unknown) |
| Grad-CAM | final + gated arm, seed 42 | Reviewer 2 |
| External (optional) | your data in `MyDrive/DR_Research/external` | Reviewer 2 (generalisation) |
| Summary | `summarize_all.py` | final-model QWK/Acc/F1 per experiment vs paper |

`SPLIT_MOVE`: `c0p0` (default, manuscript-count split used before), `c<k>p<i>` (i-th validation image of class k moves to train), `none` (original 2928/367/367). Test IDs are always the frozen reference IDs.

Paper target (final model): QWK 0.9186 +- 0.0010, Acc 85.47 +- 0.31, F1 0.703 +- 0.009.

Not covered: the original checkpoints and historical split IDs do not exist, so exact reproduction is not guaranteed; no external dataset is bundled.
