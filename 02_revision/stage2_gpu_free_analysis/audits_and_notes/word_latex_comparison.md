# Word (submitted) vs `0_main.tex` vs `revised_manuscript_draft.tex` (v1) vs v2

Sources: `v2_work/2020440-SECRETARY-R0-V1-...txt` (text dump of the submitted docx; tables flattened), `0_main.tex`, v1, v2.
Example files (`20250706_Response2reviewers.docx`, `proper sentences.docx`, `BSPC-D-25-04358_manuscript_marked.pdf`) were used for **layout style only**; their results/references belong to another study and nothing was transferred. The two PDFs could not be parsed (no PDF library); not read.

## 1. Structural / content differences

| Item | Submitted Word | `0_main.tex` | v1 | v2 |
|---|---|---|---|---|
| Split counts | 2,929 / 366 / 367 | 2,928 / 367 / 367 (matches `results/split_manifest.json`) | same as tex | same |
| Test-set claim | "test set withheld entirely… evaluated only once per configuration" | struck out; replaced by disclosure that test QWK ranked model families | same | same + new protocol paragraph (Model-Selection Protocol) |
| Gate-collapse text | Present as established finding ("Equal-Weight is Superior", 100 % label agreement) | Struck out / labelled "Unverified" | same | same |
| Grad-CAM paragraph | Present (claims Grad-CAM results) | Struck out; states no Grad-CAM was run | same | same |
| Class-wise & referable-DR paragraph | absent | absent | added (sens 0.964, spec 0.947, Severe recall 0.298) | unchanged |
| Limitations: test-set selection | absent | absent | added | unchanged |
| Calibration claims ("better-calibrated", "improve probability estimates") | present | present | present | **removed/neutralised** |
| "Full system achieved most favourable QWK" | present | present | present | **replaced** (no per-sample/full-system test evidence; Fusion CNN–ViT 0.923 > 0.9186) |
| Table 5 (efficiency) Full system | Train 142.0M / Total 142.0M | same | same | Train 27.8M; Total 142.0M; note on unreconciled teacher counts |
| AdamW citation | not cited | not cited (`loshchilov2019` in bib but unused) | same | `\cite{loshchilov2019}` added |
| AI-use statement | Yes (Author statement: Claude used solely for grammar checking) | **absent** | absent | **not inserted** (see `journal_compliance_audit.md`) |
| Conflict of interest | Yes | Yes | Yes | Yes |
| Ethics statement | "Not required, APTOS anonymised" | absent | absent | absent (needs author decision) |
| Plagiarism check statement (iThenticate) | Yes | absent | absent | absent |
| Funding | – | "No specific funding" | same | same (author must confirm) |
| Data availability | – | absent | absent | added (dataset public; code/manifest release flagged to confirm) |
| Acknowledgements | – | "…clinical experts who contributed…" (unverified, looks template-like) | same | unchanged, flagged |
| Reference format | numbered, journal style, 46 entries | natbib/`cas-model2-names`, 49 bibitems | same | 8 metadata corrections (see `reference_audit.md`) |

## 2. Reference-list differences (automated title match, **approximate**)
- tex has 49 bibitems, Word 46 numbered entries. Auto-matching failed for several entries only because of title quoting (IDF, Dosovitskiy, APTOS, Tian, Yilmaz, Macsik – the same works exist in both).
- **Genuinely different:** Word [40] Oh et al., *Sci. Rep.* 11:1897 (2021) vs tex `oh2022` = Kobat et al., *Diagnostics* 12:1975 (2022). Different papers; the tex Table 4 row ("Patch-division DenseNet, 84.90 %") matches the Diagnostics paper, so the tex entry is right in substance but had wrong authors (fixed in v2). Word [10] Serener & Serte, [11] Patil et al., [33] Chilukoti et al. are not in tex; tex has `loshchilov2019`, `zhou2012`, `kuncheva2014` not in Word. Which list is the submitted truth must be decided by the authors.
- Word [44] cites Yang et al. as AAAI 39(9) 9148–9156; tex cited the arXiv version → v2 uses AAAI. Word [42] cites HDKD as EAAI 138, 109430; tex cited arXiv → v2 uses EAAI. Word [39] Handoyo & Kusuma 881–887 and Word [38] Nabavi 2026 were correct; the tex entries were wrong (fixed).

## 3. Tables and figures
- Word: Tables 1–5, Figures 1–2; Fig. 2 = confusion matrix of seed 62 (per Word text). tex: `fig:architecture`, `fig:confusion_matrix`, five tables with labels `tab:backbones, ensemble, ablation, sota, efficiency`. Which seed the tex confusion matrix shows is not stated in the tex; recorded evidence available for all three seeds only as CSV (see consistency review). **Not verified** that the image `figures/confusion_matrix.png` equals the Word figure.
- Teacher 1 row in Table 1: Word/tex 64.85 % acc (val?) / 63.22 % / QWK 0.640 – identical in both; see `scientific_consistency_review.md` §1 for conflicts.
- Metrics (final model 0.9186 ± 0.0010, acc 85.47 ± 0.31, macro-F1 0.703) identical in Word and tex.

## 4. What the example documents contributed (format only)
Response letter layout: "General Comments" → each reviewer comment (italic) → author response (bold) → revised-manuscript excerpt in a shaded box. Used in `response_to_reviewers_draft_v2.md`.
