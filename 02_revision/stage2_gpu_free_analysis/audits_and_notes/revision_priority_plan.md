# Priority plan (deadline 18 Oct 2026)

## GPU-free work still doable now
1. Check bibliography entries/DOIs (R1.18); write AI-use statement (R1.20); proofreading (R1.21).
2. Reconcile param count (recount from notebook model definitions on CPU) and fix Table 5 (R1.13).
3. Soften superiority text vs Fusion CNN-ViT (R1.5) in Abstract/Conclusion.
4. Finish Exp4/Exp7 archived analysis (optional, historical only).
5. Read `Round1\20250706_Response2reviewers.docx` and mirror format; compare submitted docx vs tex.

## GPU work (ordered; RTX 3050 4 GB, Teacher 2 ≈ 25–30 min/epoch)
1. Fix resume script (save wait counter, best_qwk ordering, RNG) → finish Teacher 2 (or obtain server checkpoints – fastest path).
2. Student seeds 42/52/62 with saved per-sample logits/probabilities (unlocks R1.1, 1.7, 1.8, 1.14, 1.16).
3. Ablations: loss components (R1.4), T2-only vs T1+T2 (R1.6/R2.3), normalization (R1.9), augmentation (R1.10/R2.2), data fractions (R1.10).
4. Grad-CAM (R2.5) + full-system latency (R2.4) – inference only, cheap once checkpoints exist.
5. External validation (R2.1) – needs dataset download.

## Blockers
- Checkpoints missing; teacher results inconsistent; no external data; ~8 days left → consider asking the editor for an extension.
- Practical tip: ask the school-server admin for the three student/two teacher `best_model.pth` files; this removes ~2+ days of GPU time.
