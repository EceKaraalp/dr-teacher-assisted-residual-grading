Archived final-checkpoint verification (no training)

Run from the repository using the Python environment with torch, torchvision,
timm, numpy, pandas, scikit-learn and Pillow installed:

python revision_experiments\verify_archived_final.py --archive D:\m1\Documents --output results\archived_final_verification_2026-10-16

The output directory must not already exist. Archive files are read-only.
The tool uses weights_only=True and strict=True. It disables pretrained
downloads at construction because every model tensor comes from the checkpoint.
Only the three relevant archived class definitions are extracted from the
notebook; training/data-loading notebook cells are not executed.

The tool checks test-image cache pixels against all 367 resized source images,
CSV/cache label order, reconstructed test IDs against the post-hoc manifest,
then evaluates each checkpoint using the archived training-builder settings.
It also evaluates student-only and two distinct equal-fusion interpretations.
These are inference counterfactuals using the SAME trained weights; they are
not separately trained equal-fusion models or controlled training ablations.
Embedded teachers may differ from independently saved teacher checkpoints.

Outputs include local SHA-256 provenance, per-image probabilities/predictions,
confusion matrices, accuracy, macro F1, QWK, weighted one-vs-rest AUC, multiclass
Brier score (sum across classes), NLL, and ECE (15 equal-width bins).
No original-server hash identity, training-time split log, patient independence,
clinical validation, GPU latency, or statistical superiority is established.
Do not update manuscript claims until the output comparison has been reviewed.
