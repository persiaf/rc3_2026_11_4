# Independent figure quality review

Date: 2026-11-04. Reviewer: delegated paper architect/reviewer under `paper-figure` and `experiment-result-to-claim`. Verdict: **PASS for the current pilot report**, with no blocking visual or numerical issue. This is not confirmation of a paper superiority claim.

Reviewed `paper/FIGURE_PLAN.md`, `CAPTIONS.md`, `latex_includes.tex`, plotting scripts/style, both final PNGs, and Poppler-rendered complete pages of `learning_curve.pdf` and `quality_time.pdf`. PDF renders were inspected in memory, with no output or experiment regeneration. `pdffonts` confirms embedded Times New Roman/STIX CID TrueType text; `pdfimages -list` reports no embedded raster images in either PDF. Thus plotted text and marks are vector output. PNGs are generated at300dpi.

| Artifact | Findings |
| --- | --- |
| F1 learning curve | Correct line-plot choice for updates0/128/256/384/512. Raw and shared-postprocessed costs are separate panels; initialization references match the data. The best128 checkpoint and later deterioration remain visible. No invented error bars or stability implication. |
| F2 quality/time | Appropriate scatter comparison of meanJ and complete-time median. All seven baselines/controls appear; separate cold/hot panels and units are clear. Final label offsets separate4/5 and6/7. Number/shape coding preserves identity without relying only on color. Values match independently recalculated24 scene-averaged timing summaries. |
| T1 MD/TeX tables | Full Markdown table includes L/K/J/rawJ, both median/p95 pairs and fallback; compact TeX table reports a subset of the same numbers. Values agree with analysis/report rounding. When reused in a paper, retain a caption stating24 development scenes, two within-scene repetitions, CPU timing scope and selection reuse. |

Visual checks passed: serif text is readable at the planned0.95-textwidth full-width size, no clipping or label/legend collision, no internal plot titles, no decorative backgrounds, axes include equivalent-meter or millisecond units, and marker shapes/numbers allow grayscale interpretation. Legends are outside scatter data or occupy empty learning-curve regions. Figures should remain full width as the provided snippets specify; compressing each whole two-panel figure to a single column would need a fresh readability check.

Captions correctly identify the48 reused selection scenes, one training seed, and absence of independent testing or cross-seed evidence. Quality/time statistics are described as marginal summaries; timing repeats are averaged within scene. Chinese caption explicitly limits common postprocessing to insertion/prior/initial/RL and retains native full Fixed3/Detour pipelines. Learning LaTeX caption now defines J=L+0.75K and equivalent-meter units. These two caption refinements requested during review have been applied.

Fairness and claim scope are adequate: whole deployment preparation through final validation is counted; model load, input generation, serialization and separate raw-plan research audits are disclosed as excluded. Initial network and rule prior controls remain visible. Missing CSP/BC/aggregate ablations are acknowledged as unperformed; this pilot figure set must not be used to imply their results, mechanism causality, scale generalization or practical superiority. No further figure or experiment is required for closing this capped report.

Reproducibility: one source script per plot reads saved JSON; shared style and PDF/PNG pairs are present. Exact hashes of inspected figures/captions are retained in `paper/review-traces/experiment-result-to-claim/2026-11-04_run01/independent_recalculation.json`.
