# Limitations and Responsible Use

[Overview](../README.md) · [Evaluation](MODEL_EVALUATION.md) · [Handover](HANDOVER.md)

## Data limitations

- The source is a historical Sephora catalog/review snapshot, not a representative sample of all skincare customers, brands or markets. Reviews are observational, self-selected feedback rather than controlled evidence.
- Positive ratings are a proxy for preference. No exposure log tells us which unreviewed products a user saw or rejected; missing interactions are not confirmed negatives. High positive-label prevalence and reviewer selection can bias learned affinities.
- Preprocessing retains only the latest user-product review. Earlier revisions and longitudinal changes within a pair are not modeled. The processed history is not a full stream of browsing, purchasing or repeated use.
- Profile attributes may be self-reported, missing, inconsistent or coarse. Skin-type/tone row coverage is not a guarantee of accurate current profiles. Deterministic inference resolves conflicts but cannot establish which value is correct.
- The product metadata snapshot is not time-versioned. Historical prices, ingredients, names or inventory at each evaluation date are unavailable. Metadata-only content modeling does not eliminate this temporal metadata limitation.
- Interaction sparsity is about 99.908% across reviewed products, with a median of one interaction per user. A small eligible cohort drives the primary offline results.

## Evaluation limitations

- The global temporal split is more faithful to past-to-future use than a random split, but only one outer period is used. The results do not establish performance across seasons, retailers or repeated future cohorts.
- Primary metrics apply to 12,756 servable users, not all 503,216 users. Users need at least two positive training interactions and reachable future relevance. The remaining population requires separate assessment.
- Of positive future interactions for eligible users, 55.189199% concern products outside the training catalog. They are reported separately and excluded from primary relevance denominators. Reported recall is therefore reachable-catalog recall, not recall across all future preferences.
- Ranking evaluation excludes all training-seen items and uses binary rating relevance. It does not measure repeat purchase, satisfaction, suitability, retention or conversion.
- Hybrid weights were selected on inner validation only, but outer results were used to compare architectures and select the champion. A fresh holdout would be needed for an independent post-selection estimate. No uncertainty intervals or significance tests establish that a measured difference will persist.
- The hybrid's weaker later performance indicates a generalization gap relative to Model 3. Temporal drift is a plausible explanation, not a demonstrated causal attribution.
- Offline standalone metrics do not evaluate the full production routing policy, inventory/budget-filtered experience or full-history serving instances. No online A/B test has been run.

## Model limitations

- Collaborative filtering cannot infer similarities for products with no positive co-occurrences, and the current candidate policy excludes products with no historical interactions. The 69 cold catalog products are metadata-queryable only. Future-only items need an explicit new-item strategy.
- Significance shrinkage reduces rare-overlap similarity but does not remove popularity bias, review-selection bias or every spurious association. Rating-five weight 2 versus rating-four weight 1 is a fixed preference assumption.
- Content similarity reflects shared metadata terms, including branding/marketing language. It does not guarantee preference or ingredient compatibility. The implemented content fallback still uses the historical candidate catalog.
- Model 2 underperformed popularity on the frozen cohort. Its profile-based serving role enables an interpretable cold-start experience; it is not evidence of superior new-user outcomes.
- Model 4's extra components and validation-selected weights did not beat collaborative on outer test. It remains an experiment and must not be described as the production engine.
- All strategies use historical signals. They do not learn from new requests automatically, model changing preferences online, or optimize diversity/fairness. No MMR layer is implemented.
- Explanations describe how a score was assembled. Associations, TF-IDF terms and profile support counts do not explain why a product causes an outcome.

## Product limitations

- Price and stock metadata are historical snapshots. `in_stock_only` excludes explicit out-of-stock flags but retains unknown stock; it is not a live inventory promise. Missing prices cannot satisfy a requested budget.
- Category matching is exact after whitespace/case normalization, with no semantic or fuzzy search. Strong constraints can legitimately return few or zero recommendations.
- The UI does not supply shopping links or actual product images because those fields are absent from the API contract. Product placeholders are visual identifiers, not photographs.
- There is no account system, purchase flow or saved feedback loop. A known user ID is manually entered for an authorized history-based demo; normal cold-start exploration needs no ID.
- Accessibility features and responsive layouts are implemented and covered in part by tests, but this is not a comprehensive assistive-technology certification or cross-browser accessibility audit.

## Serving limitations

- Models live in a locally persisted joblib bundle, loaded into each application instance. There is no artifact registry, database, background refresh or distributed cache.
- The loader checks exact serialization dependency versions. Even a Python patch-version change can require a rebuild. A compatibility check cannot prove numerical correctness for arbitrary artifacts.
- Joblib/pickle-style artifacts can execute code on load. Only trusted local artifacts are acceptable; never load user-supplied bundles. Validation after deserialization does not make an untrusted file safe.
- No authentication, rate limiting, online monitoring, production deployment, incident alerting or load-testing platform is implemented. CORS does not provide authentication or prevent nonbrowser access.
- Reported service latency is a small local sample excluding network transport and bundle loading. It does not predict throughput, multi-worker memory use or tail latency under concurrent load.
- Rebuilding an artifact does not update already-running instances. Operators must restart deliberately and manage artifact/environment consistency.

## Safety / skincare limitations

GlowGuide is for product discovery, **not medical advice or diagnosis**. It makes no claims of clinical effectiveness or individual medical suitability. Historical preference affinity cannot establish that a product is safe or effective for someone's skin.

There is no allergy checker, ingredient contraindication engine, medication/pregnancy interaction assessment, image-based skin diagnosis, or inference of demographics/skin tone from imagery. Product ingredient text is a similarity feature, not a safety analysis. A high score is not a calibrated probability of benefit. Do not turn explanation wording into efficacy, causal or safety claims.

## Privacy considerations

- Historical user IDs are identifiers, not authorization credentials. Knowing an ID currently permits requesting history-based recommendations and explanations. This must be addressed before any public multi-user service.
- A bundle contains per-user histories/profiles; recommendation explanations can reveal products associated with that history. Access to raw CSVs, processed data, artifacts and request logs should be restricted appropriately.
- Keep real historical IDs out of committed documentation, browser fixtures, screenshots and presentations. Use `<KNOWN_DEMO_USER_ID>` as a placeholder and obtain an authorized local identifier separately.
- The app has no implemented consent, retention, deletion, identity verification or privacy-audit workflow. Dataset access does not automatically establish permission for every downstream use. Review source terms and applicable obligations before broader use.
- Public frontend environment variables are visible to clients. Never put secrets in `NEXT_PUBLIC_` configuration. Avoid logging request bodies containing identifiers/profiles without an explicit operational need and policy.
