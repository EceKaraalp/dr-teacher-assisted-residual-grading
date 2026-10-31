# Model checkpoints

Model weights are too large for ordinary git history (GitHub blocks files over 100 MB) and are not committed.
They are published as **GitHub Release assets** (up to 2 GB per file) or an external archive; the link is given here once uploaded.

**Download link:** _to be added by the authors_

## Expected contents
Per seed (42, 52, 62), per experiment arm, a `best.pth` (selected by validation QWK) together with its `config.json`, `metrics.json` and `val_logits.npz` / `test_logits.npz`:

- `teacher1` (CNN branch, class-weighted cross-entropy)
- `teacher2` (ViT branch with the learned image-wise gate; Teacher 1 frozen)
- `student/joint_*` (ConvNeXt Tiny student trained through the gated residual score)
- `student/student_ce` (independently trained CE student baseline)
- round-2 arms: `t2indep`, `nosn`, `aug`, `fusion`

The directory layout in the archive mirrors the run directories in `02_revision/stage3_round1_99job_matrix` and
`02_revision/stage4_round2_30job_matrix/results/colab_reviewer_round2_results` (for example `reference/raw_f100_s42/teacher1/`).

## Integrity
Provide SHA-256 hashes next to each uploaded archive (`certutil -hashfile <file> SHA256` on Windows). Round-2 reference runs were verified against round-1 hashes; see `reference_manifest.json`.
