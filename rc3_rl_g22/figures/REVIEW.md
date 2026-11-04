# G2.2 figures: independent reviewer check

Reviewer: delegated paper architect/reviewer, 2026-11-04. Read the two generation scripts, shared plot style and LaTeX captions; visually inspected both PNGs with view_image. No plotting or planning run was needed for this review.

Verdict: suitable for the scoped development-feedback report, with a recommended caption clarification below. No figure supports a PPO increment, unseen-scene generalization or a stable timing advantage.

## quality_time_returns

- Six quality/time points agree with independently recomputed mean per-scene percentage cost reduction and median resident-model complete time. The mean gain and median time are deliberately different summaries; the caption identifies both.
- The right panel uses paired returns against the actual same-scene root STOP, not an assumed zero STOP reward. Fixed3 and original Detour are correctly omitted from the shared-return comparison. The original versus G2 action-space distinction is explicit.
- Case IDs are categorical development cases; connected lines help trace each method and should not be read as a training or scaling trend. The caption correctly states 12 previously used cases, 72 single timing observations, excluded model loading and included instrumentation overhead.
- Axis labels, symbols and font are readable; legends do not cover observations. The color and marker/line differences distinguish methods in addition to color. No internal title is present. Vector PDF output and generation script are available.

## action_outcomes

- The 28 plotted non-STOP proposals equal 13 continued-CE plus 15 CE+gain proposals. Legend counts queried/unqueried/unseen are 4/4/5 and 7/7/1, agreeing with independent evidence recomputation. The STOP actions are correctly excluded.
- x is immediate step reward and y is the actual frozen-policy remaining return; the caption correctly rules out optimal Q and original depth-2 labels and acknowledges within-episode dependence. Some points overlap, so visible marker count is not a sample count; the explicit n values are useful.
- The relocated two-column legend sits below the data without overlap or clipping. Axes are readable at the supplied size; print review should preserve sufficient width. Model identity has distinct circle/triangle symbols. Evidence classes rely on color, so a grayscale-only export would benefit from additional class-specific fill or edge styles; this is a presentation improvement, not a data error.
- Recommended additional caption sentence: **A negative suffix may still outperform stopping at the same state; the zero line is not a STOP advantage boundary.** The saved case 8 CE+gain example has suffix −0.090335 but paired root-STOP gain +0.065059. This clarification prevents treating every lower-half point as a bad continuation decision.

## LaTeX and scope

The performance figure uses a full-width figure* and 0.95 textwidth. The action figure uses figure with 0.8 textwidth, suitable for the present single-column thesis context. For a future two-column venue, use figure* or a column-relative width to avoid overflow; no venue-specific submission readiness is claimed. Both captions identify the frozen-policy, partial-coverage evaluation scope. Figure inclusion text alone is not a compiled-paper layout check.

## Final editorial follow-up

The executor adopted the suggested negative-suffix/STOP clarification, changed the action include to a full-width figure*, and added open versus filled/black-edged markers for evidence classes. The revised PNG was regenerated from the same records and visually checked by the executor; no data or planning changed.
