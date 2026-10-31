# Change log v2 (`revised_manuscript_draft_v2.tex`, copy of v1; originals untouched)

All edits marked `\hlcyan`/`\sout` unless bibliography.

| # | Location | Change | Evidence |
|---|---|---|---|
| 1 | Setup → Data Partitioning | New paragraph "Model-Selection Protocol and Its Consequence" | forensic audit |
| 2 | Stage 1 backbone paragraph | Note that Fusion CNN–ViT 0.923 > proposed 0.9186; no superiority claim | Table 1 |
| 3 | Optimization | `\cite{loshchilov2019}` after AdamW | reference audit (uncited bib entry) |
| 4 | Efficiency section + Table 5 | Full-system trainable 142.0M → 27.8M; note on 141,999,859 / 114,175,405 frozen and unreconciled teacher counts; latency caveat | `test_metrics.json` |
| 5 | Discussion (Quantitative Evidence) | "calibration-oriented benefit" → unmeasured/hypothesised | no calibration data |
| 6 | Conclusion | Removed "full-system most favourable QWK/accuracy"; "better-calibrated" neutralised; "favorably relative to baselines" → "comparably… no paired test" | Table 1, no full-system results |
| 7 | Practical Implications | Struck calibration-improvement claim; stated calibration not evaluated | no ECE |
| 8 | Back matter | Added Data availability (flagged for author confirmation) | — |
| 9 | Table 4 | "Oh et al." → "Kobat et al."; "Minarno et al." → "Handoyo and Kusuma" | Crossref |
| 10 | Bibliography | cao2020 & liu2020 DOIs; oh2022 authors; minarno2022 authors/pages; nabavi2024 vol/year/names; yang2025 → AAAI; elassiouti2024 → EAAI | Crossref |

Not changed (flagged): AI-use statement (unverified), Acknowledgements sentence, ethics statement, Abstract (not in tex), "consistent improvements over standalone backbones" sentence in Discussion, teacher param counts, Teacher 1 QWK narrative, preamble duplicates.
Not compiled (no LaTeX engine).
