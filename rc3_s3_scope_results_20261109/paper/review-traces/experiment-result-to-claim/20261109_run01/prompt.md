# Independent paper architect/reviewer prompt

Assigned by the parent executor to a separate reviewer agent on 2026-11-09.
Use experiment-result-to-claim and judge saved evidence independently; do not accept the executor's intended conclusion.

Intended claim: DRL trained in large complex 10x10 parent scenes, frozen after training, provides a positive planning-quality increment over both matched supervised controls and rapidly solves different-size local tasks with complete navigation, coverage, same-arm reachability and safety checks. The frozen protocol operationalizes this as six quality cells and twelve deployment strata with the original AND rules. Supervision, rules and search remain initialization, auxiliary processing or controls; they must not replace the fixed DRL research goal.

Evidence and provenance:

- Approved original v3 protocol and registry, separate original S3 authorization, INPUT_SCOPE_AMENDMENT.json and RECOVERY_DECISION.json.
- Source binding 311a58bd867e95c852ec3246c54a398468eb526db391fbb6d12239c6a3379949.
- Three fixed optimizer seeds 11622026/11622027/11622028, 108 declared cases, seven methods, one separate quality request and three cold plus three hot complete requests per case/method/seed.
- REQUEST_METRICS.csv, SCIENTIFIC_ANALYSIS.json, RESOURCE_ACCOUNTING.json, REQUEST_STATE_INDEX.json, RESULT_ARTIFACT_INDEX.json, MATCHED_COMPARISONS.json, METHOD_QUALITY_TIME.csv, TIMING_BY_REVISION.csv, CONSTRAINT_SOURCE_AUDIT.json, MODEL_SOURCE_AUDIT.json, and production saved artifacts.
- Retain 2057 inherited successes and two historical failures. scope_r1 may execute only the other 13817 IDs; no refill of either old failed slot.

Required independent verification:

1. Actual new versus inherited requests/resources and the two unfilled old failures.
2. Independent quality/cold/hot completeness, six quality and twelve deployment cells, original AND thresholds.
3. Independently reaggregate complete quality cells by actual physical parent and matched seed, bootstrap 20000 PCG64 parent-cluster resamples at seed 11632026, and independently recompute complete median/p95 timing and ratios.
4. Check index, saved raw routes and CSV; RL_raw and RL_plus_H must share the same pre-H raw route. Attribute H gains separately.
5. Statically hash all 36 fixed model provenance files, bound execution sources and frozen evidence; no model deserialization.
6. Treat 6 seen and 12 unseen physical parent layouts as clusters. Requests/repeats are not independent layouts. Report version-mixed timing and scope_r1 scientific admission limitations.
7. Explicitly judge whether actual advantages over supervision and strong rules, across sizes, and the rapid-planning goal are supported. Do not add a new performance prerequisite.

Authority: read saved results, recompute statistics and hash sources only. Scientific requests, model/scene generation, navigation/native/H/optimizer calls and original60 semantic reads are prohibited. Do not change execution code, old evidence, Git or other user files. Only write INDEPENDENT_REVIEW.md/.json and this dedicated trace directory. Record actual audit time and zero scientific calls.

Return structured claim_supported yes|partial|no; what_results_support; what_results_dont_support; missing_evidence; suggested_claim_revision; next_experiments_needed (proposals only); confidence high|medium|low; technical audit GO|REVISE|BLOCKED; full-grid scientific status. If paper/PAPER_CLAIM_AUDIT.json is absent, mark paper_claim_audit unavailable and downstream paper-writing judgment provisional, without blocking this results report. Record prompt, raw reviewer response, parsed verdict, routing and pending-external-review flag.

Skill trace note: the installed skill's relative shared-references/review-tracing.md was absent at the referenced location, and rg found no copy under the installed skill root. This trace supplies all explicitly required fields from SKILL.md directly.
