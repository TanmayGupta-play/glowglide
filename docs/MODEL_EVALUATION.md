# Model Evaluation

[Overview and exact six-metric benchmark table](../README.md#recommendation-models) · [Architecture](ARCHITECTURE.md) · [Limitations](LIMITATIONS.md)

## Dataset

The source is the historical `nadyinky/sephora-products-and-skincare-reviews` dataset referenced by `scripts/download_data.py`. Preprocessing restricts products to skincare, validates required review fields and retains the latest interaction for each user-product pair.

| Processed-data measure | Value |
|---|---:|
| Skincare products in metadata | 2,420 |
| Interactions | 1,088,886 |
| Unique users | 503,216 |
| Reviewed products | 2,351 |
| Positive interactions | 893,392 |
| Strong-positive interactions | 584,915 |
| Skin-type coverage, interaction rows | 89.87% |
| Skin-tone coverage, interaction rows | 84.48% |
| Median interactions per user | 1 |
| Date range | 2008-08-28 through 2023-03-21 |

Interaction density is approximately **0.0920%**, calculated as 1,088,886 / (503,216 × 2,351), using the reviewed-product universe. Sparsity is approximately **99.908%**. The other 69 skincare products have no processed review history. Using all 2,420 metadata products would produce a different density; the denominator matters.

Most users have very little history. This motivates complementary history, metadata and profile signals and explicit cold-start routes. It does not establish that fusing all signals will improve ranking; the hybrid experiment tests that hypothesis.

## Relevance

- `positive = 1` when `rating >= 4`.
- `strong_positive = 1` when `rating == 5` and `is_recommended == 1`.
- Primary experiments use **positive**, not strong-positive relevance.

These are preference proxies from observed ratings, not purchases, exposure-adjusted outcomes or clinical suitability. Ratings below four do not create positive collaborative edges or positive content profiles. They remain seen interactions and contribute to Model 2's outcome denominators.

## Temporal split

`temporal_split()` uses the timestamp's global 0.80 quantile with one cutoff. Rows at or before the cutoff train the models; later rows evaluate them. Tied timestamps stay together, so row fractions need not be exactly 80/20. Users are not selected before splitting. A random split could let later behavior inform predictions for earlier events and would less closely reflect the intended past-to-future task.

| Outer benchmark | Value |
|---|---:|
| Cutoff | 2022-02-02 |
| Train interactions | 871,530 |
| Test interactions | 217,356 |
| Train positive interactions | 713,025 |
| Test positive interactions | 180,367 |
| Train candidate products | 1,790 |
| History-eligible users | 18,274 |
| Servable users | 12,756 |

The candidate set includes **all products appearing in training**, including products without positive interactions. It excludes test-only products. Vocabulary/IDF, profiles, similarities and affinity statistics are fitted within the appropriate training period. Content features use product metadata only, but the metadata source itself is a static snapshot rather than a time-versioned catalog.

History eligibility requires at least two positive training interactions and at least one positive test interaction. For eligible user `u`, relevance `R_u` is the set of positive test products also present in the training catalog. Users with nonempty `R_u` are servable. All training-seen products are excluded from recommendations, including negatively rated products. Offline evaluation does not apply stock, price or category filters.

## Metrics

Let `h_r` be binary relevance at rank `r`, and `H` the number of hits in the first K positions. Metrics are computed per servable user and averaged, except catalog coverage, which aggregates the recommendation union. K=10 in the frozen benchmark.

| Metric | Implemented definition | What it measures |
|---|---|---|
| Precision@K | `H / K`, even when fewer than K items are returned | Hit frequency per offered recommendation slot. |
| Recall@K | `H / |R_u|` | Fraction of reachable positive future products recovered. |
| HitRate@K | `1` if any hit, otherwise `0` | Fraction of users receiving at least one hit. |
| NDCG@K | `sum(h_r / log2(r+1))`, divided by ideal DCG with `min(K, |R_u|)` hits | Relevance with greater credit at earlier ranks. |
| MAP@K | Mean of `sum(precision at each relevant rank) / min(K, |R_u|)` | Early and repeated relevant placements. |
| Catalog Coverage@K | Distinct recommended products / training candidate count | Breadth of catalog exposure, not relevance. |

Using several metrics separates early-rank quality, user-level success and catalog breadth. Coverage alone cannot establish recommendation usefulness. Scores are low in absolute terms because the observed future positives are sparse and incomplete; unobserved products are not confirmed dislikes.

The evaluator rejects duplicate, seen, out-of-catalog or over-K recommendations. It also reports recommendation-count mean/median/full/fewer/zero diagnostics. Recommendation latency measures the `recommend()` call, not fitting or the entire experiment. No model is allowed to alter shared metric denominators to improve its results.

## Experiments

The exact frozen six-metric table is maintained in the [root README](../README.md#recommendation-models); it was checked against the local `artifacts/*metrics.json` records. The following interpretations use the same outer cohort.

### Model 0 — Most Popular

**Hypothesis:** frequently liked products provide a useful nonpersonalized reference.

**Method:** count distinct positive training users for each candidate. Rank descending by count, then ascending product ID. This is the reference against which personalization adds value.

**Result:** NDCG@10 = 0.012488504144695494; coverage = 0.01452513966480447.

**Interpretation:** the model offers broad user availability but narrow catalog exposure. It cannot express individual preferences. Its production role is the no-history/no-profile fallback.

### Model 1 — Personalized TF-IDF Content

**Hypothesis:** metadata themes in previously liked products help identify other relevant products.

**Method:** concatenate product name, brand, secondary/tertiary categories, ingredients and highlights. TF-IDF uses English stop words, unigrams/bigrams, `min_df=1`, sublinear term frequency and L2 normalization. Fit on training-catalog metadata. Build a positive-history profile with rating-four weight 1 and rating-five weight 2, normalize it and rank cosine similarity. Review text, outcome counts and popularity are not content features.

**Result:** NDCG@10 = 0.015016724422000007; coverage = 0.9329608938547486, the broadest catalog coverage.

**Interpretation:** content improves relevance over popularity and explores more products, but shared language is an imperfect preference signal. The implemented catalog policy still excludes products without training interactions, even though content methods could support a wider cold-item design in future.

### Model 2 — Skin Profile Affinity

**Hypothesis:** users with similar reported skin type/tone show useful product-affinity patterns.

**Method:** infer each training user's type and tone independently from nonmissing modal values; latest timestamp and then lexical order resolve ties. Learn positive rates from both positive and negative training rows. Smooth overall product rates toward the global rate with strength 20; type/tone rates toward product overall with strength 20; exact pairs toward the mean of smoothed type/tone rates with strength 40. Fixed component weights are overall 0.10, type 0.25, tone 0.15, pair 0.50, renormalized over supplied/inferred attributes. Empty cells equal their priors.

**Result:** NDCG@10 = 0.008641444499491528, below popularity. All five relevance metrics underperform Model 0; coverage is also narrow at 0.02346368715083799.

**Interpretation:** this is a negative ranking result. Coarse demographic/profile grouping is not enough to recover individual preference reliably. The model remains a transparent profile-based cold-start option in serving; that product role is not evidence that it beats popularity for new users, who were not the primary benchmark cohort.

### Model 3 — Item-Item Collaborative Filtering

**Hypothesis:** products liked by overlapping users provide a stronger preference signal than metadata/profile similarity alone.

**Method:** build sparse binary positive user-item matrix `B`. Let `co(i,j)` count unique positive training users shared by two products, and `s_i` count positive training users for item `i`.

```text
raw_cosine(i,j) = co(i,j) / sqrt(s_i * s_j)
similarity(i,j) = raw_cosine(i,j) * co(i,j) / (co(i,j) + 10)
similarity(i,i) = 0
score(u,c) = sum_i(weight(u,i) * similarity(i,c)) / sum_i(weight(u,i))
weight(u,i) = 1 for rating 4; 2 for rating 5
```

No positive support produces zero similarity. Significance shrinkage **10.0** is a fixed assumption, not an outer-test-tuned parameter. The sparse representation avoids a dense half-million-user matrix. Only positive candidate scores are recommended; zero-similarity items are never filler. Unknown/no-positive-profile users return no standalone collaborative recommendations. Ties use ascending product ID.

**Result:** NDCG@10 = 0.035985149669067955, Recall@10 = 0.057273285260795496 and MAP@10 = 0.024222856698542742. Model 3 leads every relevance metric, with coverage 0.66815642458100555.

**Interpretation:** overlapping positive histories provide the strongest measured ranker. This establishes the champion architecture for serving, subject to cold-start limitations. Explanations expose history contributions to the exact score.

### Model 4 — Validation-Tuned Hybrid

**Hypothesis:** complementary signals can improve generalization through validation-selected fusion.

**Method:** split **outer train only** again at the global 0.80 timestamp quantile. Fit four components once on inner train. For each user, normalize each component by its positive maximum; unavailable components become zero. Renormalize configured weights over available positive-weight signals. Fuse, exclude seen and return positive scores only.

| Inner validation / search | Value |
|---|---:|
| Cutoff | 2021-03-24 |
| Inner train / validation rows | 697,494 / 174,036 |
| Inner candidates | 1,458 |
| History-eligible / servable users | 13,549 / 9,314 |
| Coarse grid | 35 combinations, step 0.25, including pure-model boundaries |
| Local grid | 491 combinations, step 0.05, within ±0.25 per coarse winning coordinate |

Selection maximizes **NDCG**, then **MAP**, then **Recall**, with absolute tolerance `1e-12`, then the lexicographically smallest tuple in `(popularity, content, profile, collaborative)` order. Normalized inner scores are cached in a temporary disk-backed array; components are not refitted for each combination.

| Winner | Weights in fixed order | Inner NDCG@10 | Inner MAP@10 | Inner Recall@10 |
|---|---|---:|---:|---:|
| Coarse | (0.25, 0.25, 0.00, 0.50) | 0.0438442994715616 | 0.026947788264217253 | 0.0758396993673818 |
| Refined, frozen | **(0.15, 0.15, 0.10, 0.60)** | **0.04502009917146835** | **0.02746683241330315** | **0.07803188029090018** |

After freezing, refit all four components on outer train and evaluate the hybrid on outer test. Inner ablations remove a selected component and renormalize the rest, without changing the winner:

| Removed component | Δ inner NDCG@10 | Δ inner MAP@10 | Δ inner Recall@10 |
|---|---:|---:|---:|
| Popularity | -0.004628418 | -0.003125211 | -0.009564713 |
| Content | -0.001004879 | -0.001052416 | +0.000179425 |
| Profile | -0.000614854 | -0.000342552 | -0.001576963 |
| Collaborative | -0.013555019 | -0.007694018 | -0.019258267 |

Ablation deltas above are rounded; the selected metrics above are unrounded. The full local results remain in the ignored tuning/metrics artifacts.

**Result:** outer NDCG@10 = 0.030853807080895908, MAP@10 = 0.01955672419427102 and Recall@10 = 0.050030374247063768. All five outer relevance metrics and coverage are below Model 3.

**Interpretation:** refinement improved the inner selection objective, but that advantage did not translate into beating collaborative on the later outer period. This generalization gap is consistent with temporal distribution change and/or validation selection effects; it does not prove a particular cause. Additional complexity was not rewarded. Model 4 remains an experiment, not the production champion.

## Cold start

Among 18,274 history-eligible users, 5,518 have only out-of-catalog positive future products: **30.195907%**. Their exclusion from primary metrics is disclosed, not hidden. They have qualifying past history; “cold-start-only” here refers to the future item catalog.

Of 69,028 eligible positive future interactions, 38,096 (**55.189199%**) lie outside the 1,790-item train catalog, leaving 30,932 reachable interactions. This rate is not the fraction of all test rows, nor the fraction of all users. Mixed users remain servable but their outside-catalog positives are outside primary relevance denominators.

Reporting reachable ranking quality separately avoids attributing impossible catalog retrieval to ranking order. It also means these metrics cannot be presented as end-to-end recall over every future preference. New-user and new-item coverage need separate product evaluation. Production's full-history refit expands candidates to 2,351 but does not solve the 69 products with no history or future arrivals.

## Model selection

Model 3 leads the common outer relevance metrics and supports direct history-based explanations. Its lower complexity than Model 4 is useful, but measured relevance is the primary reason for selection. Serving uses deterministic routing, not the rejected hybrid: collaborative first, usable content fallback, explicit profile otherwise, then popularity.

The outer benchmark compared standalone models. It did not benchmark the combined production routing policy, business-filtered experience or full-history fitted serving instances. There is no online satisfaction or A/B-test evidence yet.

## Failure cases

- **New item:** no candidate membership/positive co-occurrences means no collaborative recommendation.
- **Insufficient user history:** most dataset users do not meet the offline history threshold; a serving fallback is needed.
- **Coarse profile:** shared type/tone can obscure strong individual preference differences.
- **Content mismatch:** similar ingredient/marketing terms do not establish that a user will like a product.
- **Temporal generalization:** inner-optimal fusion was weaker than Model 3 in the later period; future catalog/user changes may change rankings again.
- **Constraint exhaustion:** seen exclusion and filters can leave few or no positive candidates; the service returns fewer items without filler.

## Experimental integrity

Fusion weights were selected using a nested global temporal validation split inside the outer training period. **The outer test set was not used for fusion-weight selection. The outer test was evaluated only after fusion weights were frozen.** Inner validation did not fit the component models, and ablations did not revise selected weights. The collaborative shrinkage and profile smoothing policies are documented fixed assumptions.

Outer outcomes were used for model comparison and champion selection. Therefore the project must not describe the outer test as never observed, or claim an independent post-selection estimate of the final product's performance. A fresh future period or online evaluation would strengthen that claim. No statistical significance intervals or repeated random-split results are provided.

This documentation pass checked saved metrics and reran tests, not frozen experiment scripts or expensive hybrid tuning. Serving refits all historical data only after the offline architecture choice; serving artifacts and benchmark artifacts remain distinct.
