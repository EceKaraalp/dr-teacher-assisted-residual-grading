# Formatting audit

## Compile status
**Not compiled.** No LaTeX engine is installed (`pdflatex`, `latexmk`, `xelatex`, `lualatex`, `tectonic`, MiKTeX/`kpsewhich`: all absent). Also the class file `cas-dc` (Elsevier CAS) and `cas-model2-names.bst` are not in the project, and `figures/confusion_matrix.png` and `prism-uploads/TS_ConvNeXtTiny_Residual_Architecture_English_Detailed....pdf` referenced by `\includegraphics` were not found relative to the repo root. **No claim of successful compilation is made.** Checks below are static.

## `\hlcyan`
- Defined as `\DeclareRobustCommand{\hlcyan}[1]{{\sethlcolor{cyan}\hl{#1}}}` with `soul` loaded and `color` loaded (`cyan` is a base colour of `color`; `xcolor` is not needed). `\sout` needs `ulem` (loaded with `[normalem]`).
- soul limitations: `\hl` argument must not contain `\par`, sectioning commands, and is fragile with `\cite`/`\ref`. In v2 I kept `\hlcyan` on plain text only, and moved `\cite`/`\ref` outside highlights; original v1/0_main usages containing `\textbf{...}` inside `\hlcyan` were kept (existing in original). **Untested without a compiler.** Highlights spanning several paragraphs, if any, would error.

## Static checks (script over v2)
- All 8 `\ref` targets defined (`fig:architecture, fig:confusion_matrix, sec:teacher_ensemble, tab:ablation, tab:backbones, tab:efficiency, tab:ensemble, tab:sota`); `\begin`/`\end` counts equal (27/27); braces balanced.
- Tables numbered by custom `\lefttablecaption` (`\refstepcounter{table}`), order backbones → ensemble → ablation → sota → efficiency = Tables 1–5 (consistent with Word).
- 8 numbered equations, none cross-referenced by `\eqref`.
- Hard-coded "Section~4" / "Section on..." references exist; fragile if sections change.
- Preamble loads `natbib` three times with different options (`authoryear,longnamesfirst`, `authoryear`, `numbers`) and `caption` three times – may cause option-clash errors; line 1 has a commented alternative class line. Pre-existing, not changed.
- Table 5: Full-system trainable fixed (27.8M); teacher rows still "~8.2M/~113.8M" – unreconciled with the 114.2M frozen count (see consistency review §2). `Not measured` latency retained.
- The class-wise/referable-DR paragraph (v1) sits in Results after the confusion-matrix discussion – appropriate section.
- Abbreviations (QWK, KD, CE, ECE) – not exhaustively audited.
