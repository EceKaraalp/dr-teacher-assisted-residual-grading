# Reference audit

Method: (a) script cross-check of `\cite{}` keys vs `\bibitem` keys in `0_main.tex`; (b) Crossref REST lookup (external) for every DOI present, comparing title/authors/year/volume/pages. Raw outputs: `v2_work/refs.json`, `v2_work/crossref.json`.
**Status words:** "Crossref-verified" = the DOI resolves in Crossref to a record whose title, first authors and year match. Nothing else is claimed verified. **Claim-support (does the paper actually support the sentence it is cited for) was NOT checked** for any reference – it requires reading each source.

## 1. Citation ↔ bibliography
- 49 bibitems; every `\cite` key has an entry (0 missing).
- Cited-but-absent: none. Uncited bibitem: **`loshchilov2019`** (AdamW). AdamW is used in §Optimization → v2 now cites it there (`\cite{loshchilov2019}`).
- Bibliography order ≠ order of first citation (style `cas-model2-names` + `thebibliography`; numeric `natbib` option is also loaded – three `natbib` loads in preamble, see `formatting_audit.md`).
- No duplicates detected by key/title (Yang/Hinton etc. appear once).

## 2. Verified errors found (fixed in `revised_manuscript_draft_v2.tex`)
| Key | Problem (Crossref evidence) | v2 correction |
|---|---|---|
| `cao2020` | DOI 10.1016/j.patrec.2020.11.009 resolves to an unrelated speech-emotion paper. CORAL paper = 10.1016/j.patrec.2020.11.008 (PRL 140:325–331) | DOI fixed |
| `liu2020` | DOI 10.1016/j.neucom.2019.10.038 resolves to an unrelated flight-control paper. Correct: 10.1016/j.neucom.2020.01.025 (Neurocomputing 388:34–44) | DOI fixed |
| `oh2022` | DOI/title correct, authors wrong. Crossref: Kobat, Baygin, Yusufoglu, Baygin, Barua, Dogan, Yaman, Celiker, Yildirim, Tan, Tuncer, Islam, Acharya (*Diagnostics* 12(8):1975) | authors fixed; in-text "Oh et al." → "Kobat et al." (Table 4) |
| `minarno2022` | Authors are Handoyo & Kusuma; pages 881–887 (bib had 927–933) | fixed; in-text name fixed |
| `nabavi2024` | Medical Physics 53(2), e70350, published **2026** (bib "2024"); first-name form "Anvari Hamedani" | fixed (key name left as `nabavi2024` to avoid breaking cites – rename optional) |
| `yang2025` | cited as arXiv DOI 10.48550/arxiv.2502.18510 (Crossref 404). Published: AAAI 39(9):9148–9156, DOI 10.1609/aaai.v39i9.32990 (Crossref-verified, authors match) | replaced with AAAI |
| `elassiouti2024` | arXiv DOI (Crossref 404). Published: Eng. Appl. Artif. Intell. 138:109430, DOI 10.1016/j.engappai.2024.109430 (Crossref-verified) | replaced |

## 3. Crossref-verified, no discrepancy found
yau2012, wilkinson2003, krause2018, gulshan2016, ting2017, he2016, huang2017, zhu2024, bappi2025, zhang2025, islam2023, liu2022, farag2022, shakibania2024, tian2023, romero2024, ikram2025, azzou2025, nazih2023, goh2024, gou2021, luo2020, ju2021, moya2024, wang2024, alshafi2024, li2024, litvinov2026, liu_medvit2024, hossain2025, macsik2023, niranjan2025 (title/author/year level; volume/pages not exhaustively compared). `dosovitskiy2021` year auto-extracted as 2010 from arXiv id – false alarm.
A Crossref rate-limit (HTTP 429) interrupted part of the bibliographic search; entries not listed in §2–3 were **not externally verified**.

## 4. Entries without DOI/URL or not DOI-verifiable
`zhou2012`, `kuncheva2014` (books), `tan2019` (EfficientNet), `hinton2015`, `miyato2018` (ICLR), `yilmaz2025`, `dosovitskiy2021` (arXiv/ICLR), `idf2021`, `aptos2019` (dataset URL). Not externally verified; arXiv-only items may exist as formal versions (Yilmaz & Aiyengar should be checked).

## 5. Other flags
- Table 4 (SOTA) values were copied from cited papers; **not re-checked against the source papers** (e.g. 0.769† and 0.900‡ footnoted entries).
- Word vs tex reference lists differ (see `word_latex_comparison.md` §2) – authors must decide the final list.
- Reviewer-relevant: references related to teacher-selection, calibration and Grad-CAM are not yet in the paper (to be added only after those experiments exist).
