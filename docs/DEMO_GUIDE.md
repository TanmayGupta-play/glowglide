# Recruiter Demo Guide — 5–7 Minutes

[Overview](../README.md) · [Run instructions](HANDOVER.md) · [Benchmark](../README.md#recommendation-models)

## Prepare before the demo

1. Use the existing processed data and trusted serving bundle. Start FastAPI and Next.js using the [handover commands](HANDOVER.md).
2. Confirm `http://localhost:8000/health` reports a loaded bundle and open `http://localhost:3000`.
3. Have an **authorized** known positive-history identifier available privately. This guide uses `<KNOWN_DEMO_USER_ID>`; it is a placeholder, not a valid request ID. Do not paste real historical IDs into committed docs, screenshots or a recorded/public demo without appropriate authorization.
4. Rehearse that identifier once: choose a user with an unseen positive collaborative signal. Some valid historical users naturally take the content fallback. Do not alter routing to force a demonstration.
5. Keep the README benchmark and optional Swagger `/docs` in separate tabs. Do not rebuild data, install packages or tune models live.

## Demo sequence

| Time | Action | What to say |
|---|---|---|
| 0:00–0:35 | Show the GlowGuide landing experience. | “The problem is personalized product discovery with very sparse review histories. GlowGuide supports history-based recommendations and an explicit cold-start path, with an explanation for each pick.” |
| 0:35–1:15 | Leave user ID blank. Select **Combination** and **Medium**, keep 10 results and click **Find my skincare**. | “A new user does not need an account or a historical ID. This route uses learned historical affinity for the supplied profile.” Confirm **Your skin profile**. |
| 1:15–1:55 | Open **Why this?** on a product, then close it. | “These are smoothed preference components and support counts from historical reviews. They are not medical effectiveness claims.” Point to the returned type/tone and affinity details. |
| 1:55–2:45 | Enter `<KNOWN_DEMO_USER_ID>` privately, clearing optional filters if needed, and request recommendations. | “With usable positive history, the champion is item-item collaborative filtering. It finds products liked by overlapping users.” Confirm **Your product history**; if it says **Your preferences**, explain the legitimate content fallback. |
| 2:45–3:20 | Open a history-based **Why this?**. | “The contributing products come from the user's actual positive history. Their similarities contribute mathematically to the recommendation score. This is an association, not a causal claim.” For content fallback, show the actual matching terms instead. |
| 3:20–4:00 | Set a maximum price, for example **40 USD**, and choose **Moisturizers**. Request again; optionally open **View details**. | “Budget, category and stock constraints filter the ranked candidates. They do not change model scores or retrain the ranker. Details come from the same metadata API.” |
| 4:00–4:30 | If useful, set a restrictive budget such as **0** with a category; show the result and **Adjust filters**. | “An empty list is valid when constraints eliminate candidates. We return fewer products rather than inventing matches or switching models just to fill the page.” Zero results depend on the catalog; do not promise them before rehearsing. |
| 4:30–4:55 | Briefly show `/health` or Swagger `/docs`. | “FastAPI loads a trusted serving bundle once. The frontend only calls the API; it never accesses Python models directly.” |
| 4:55–6:00 | Show the README benchmark table and hybrid summary. | “All five experiments use the same temporal benchmark. Collaborative won every relevance metric. Hybrid weights were selected inside the training period, but hybrid underperformed collaborative on the later outer test, so it was not promoted.” |
| 6:00–6:20 | Close with one engineering boundary. | “After model selection, serving models are refitted on all available processed history. Saved held-out results remain a separate experimental record. This is a local application, not a deployed or medically validated service.” |

If time permits, remove the ID and both profile selections to show **Community favorites**. This demonstrates popularity without requiring any user history. Do not leave a private ID visible while discussing the benchmark.

## Key points to make accurately

- The history-first ranker is **Model 3**, not the experimental hybrid.
- Profile and content fallback routes are deterministic and transparent; they are not hidden weighted fusion.
- The outer test was not used to choose Model 4 fusion weights. It was used to compare models after those weights were frozen.
- The benchmark is for a servable cohort. More than half of eligible positive future interactions were outside the training catalog, and that limitation is reported separately.
- Product inventory and price are historical metadata. The UI is for discovery, not a live retailer checkout or skin diagnosis.
- Explanations reuse model data. No LLM invents explanation text, product terms or safety claims.

## What NOT to spend time on

- Walking through every preprocessing line, metric formula or dependency during the product demo; link the technical documents for follow-up.
- Running the million-row build or hybrid grid search live.
- Showing raw user IDs, CSV rows or a directory full of artifacts.
- Highlighting raw model scores as if they were calibrated match probabilities.
- Apologizing for a legitimate empty state or hiding the weaker Model 2/Model 4 results.
- Claiming deployment, online A/B results, clinical validity, authentication or live stock checks that do not exist.

## Recovery plan

If the service is unavailable, show the UI's friendly error and check `/health`, bundle presence and the configured API URL. Follow [troubleshooting](HANDOVER.md#troubleshooting); do not fabricate recommendations. If a historical user has no collaborative signal, explain the content fallback or use a separately authorized, rehearsed identifier. If filters leave a short list, keep that result visible and explain the no-filler policy.
