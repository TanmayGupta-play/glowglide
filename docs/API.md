# API Reference

[Overview](../README.md) · [Architecture](ARCHITECTURE.md) · [Operations](HANDOVER.md)

The implemented contracts are [schemas.py](../src/glowguide/api/schemas.py) and [main.py](../src/glowguide/api/main.py). Development base URL: `http://localhost:8000`. Interactive Swagger: `/docs`; ReDoc: `/redoc`; OpenAPI schema: `/openapi.json`. No authentication is implemented.

## Configuration and startup

From the repository root, with dependencies and a trusted bundle installed:

```sh
python -m uvicorn glowguide.api.main:app --app-dir src --reload
```

| Variable | Default | Behavior |
|---|---|---|
| `GLOWGUIDE_BUNDLE_PATH` | `artifacts/serving_bundle.joblib` | Relative paths resolve against the project root, not the working directory. Absolute paths are also accepted as trusted operator configuration. |
| `GLOWGUIDE_CORS_ORIGINS` | `http://localhost:3000` | Comma-separated origins, trimmed. CORS permits GET/POST and Content-Type; credentials are disabled. |
| `NEXT_PUBLIC_GLOWGUIDE_API_URL` | `http://localhost:8000` | Frontend variable, not an API setting. Configures browser requests. |

FastAPI loads one bundle during each application lifespan. Missing, corrupt, version-incompatible or dependency-incompatible bundles fail startup; the app does not silently refit or run degraded. `create_app(service=...)` supports an injected service for tests.

## GET /health

Returns HTTP 200 after successful startup:

```json
{
  "status": "ok",
  "bundle_loaded": true,
  "bundle_version": "1",
  "candidate_products": 2351,
  "historical_users": 503216
}
```

Fields are `status` (literal `ok`), `bundle_loaded` (boolean), `bundle_version` (string), and the two integer counts. Counts above correspond to the current full-history bundle; they can change after a data rebuild. This is an artifact-readiness check, not a live inventory or external-dependency health test.

## POST /recommendations

Send `Content-Type: application/json`. An empty object is valid and uses defaults.

| Request field | Type | Default / constraint |
|---|---|---|
| `user_id` | string or null | null; optional historical identifier. |
| `skin_type` | string or null | null; normalized using preprocessing's shared rule. |
| `skin_tone` | string or null | null; normalized using the same rule. |
| `max_price` | finite number or null | null; must be ≥ 0; compared to `price_usd`. |
| `category` | string or null | null; exact case-insensitive category match after whitespace normalization. |
| `in_stock_only` | boolean | true. |
| `top_k` | integer | 10; strict integer from 1 through 50; booleans/fractions are invalid. |

Unknown fields are forbidden. Empty strings are trimmed to null where appropriate. Profile normalization converts `Combination` to `combination` and `light medium` to `light_medium`, including CamelCase and separator normalization. Profile strings are **not an enum**: unseen groups back off through the learned smoothing priors, rather than returning a schema error.

Example profile-only request:

```json
{
  "user_id": null,
  "skin_type": "Combination",
  "skin_tone": "Medium",
  "max_price": 40,
  "category": "Moisturizers",
  "in_stock_only": true,
  "top_k": 10
}
```

### Response envelope

| Field | Type / meaning |
|---|---|
| `strategy` | `collaborative`, `content_fallback`, `skin_profile` or `popularity`. |
| `requested_top_k` | Integer request limit. |
| `returned_count` | Number of recommendation items. May be zero or less than K. |
| `user_history_available` | Boolean: user has positive historical interactions. |
| `skin_profile_used` | `{ "skin_type": string or null, "skin_tone": string or null }` for profile routing; otherwise null. |
| `filters_applied` | Object with `max_price` (number/null), `price_field` (literal `price_usd`), `category` (normalized string/null), `in_stock_only` (boolean). |
| `candidate_pool_size` | Positive unseen candidates before business filtering, not catalog size or requested K. |
| `bundle_version` | String artifact version. |
| `recommendations` | Ordered array of the items described below. |

The following is a **synthetic schema example**, not a real product or measured production response. It shows the complete popularity response shape:

```json
{
  "strategy": "popularity",
  "requested_top_k": 1,
  "returned_count": 1,
  "user_history_available": false,
  "skin_profile_used": null,
  "filters_applied": {
    "max_price": null,
    "price_field": "price_usd",
    "category": null,
    "in_stock_only": true
  },
  "candidate_pool_size": 1,
  "bundle_version": "1",
  "recommendations": [{
    "product_id": "EXAMPLE_PRODUCT",
    "product_name": "Example moisturizer",
    "brand_name": "Example brand",
    "primary_category": "Skincare",
    "secondary_category": "Moisturizers",
    "tertiary_category": null,
    "price_usd": 25.0,
    "out_of_stock": false,
    "rating": 4.2,
    "score": 12.0,
    "score_type": "positive_user_count",
    "explanation": { "type": "popularity", "positive_user_count": 12 }
  }]
}
```

### Routing and score types

| Strategy | Selection | `score_type` / raw score |
|---|---|---|
| `collaborative` | Positive history with at least one unseen positive collaborative candidate. | `adjusted_collaborative_affinity`: rating-weighted average of shrunk item similarities. |
| `content_fallback` | Collaborative pool empty, but the positive-history TF-IDF profile has positive signal. | `tfidf_cosine_similarity`: user profile–product cosine. |
| `skin_profile` | History-based signals unusable and explicit type and/or tone supplied. | `smoothed_skin_profile_affinity`: existing Model 2 weighted affinity. |
| `popularity` | No usable history signal and no explicit profile. | `positive_user_count`: distinct positive historical user count. |

Scores across strategies are **not calibrated probabilities** and should not be compared as a common confidence scale. A supplied profile does not override usable collaborative/content history. Model 4 does not participate in serving.

### Ranking and filters

Rank positive raw scores descending, with ascending product ID for exact ties. Exclude **all** historical seen products for recognized users, including negative-only history. Apply business filters to that ranking, then take Top-K. No filter retrains a model or alters a score, and no strategy switch occurs merely because filters remove every result.

- `in_stock_only=true` removes products explicitly marked out of stock. Unknown stock remains eligible; the flag does not certify live availability.
- Category matches any of `primary_category`, `secondary_category`, `tertiary_category`, ignoring case and normalizing whitespace. No fuzzy matching.
- An explicit budget requires a nonmissing `price_usd <= max_price`. Prices retain the dataset's USD semantics; no conversion.
- No zero-score filler or duplicates. The full positive unseen catalog pool is considered.

### Recommendation items and product metadata

Each recommendation contains all `ProductMetadata` fields plus finite numeric `score`, the `score_type` literal and one discriminated `explanation` object. Optional metadata is null, never NaN/Infinity.

| ProductMetadata field | Type |
|---|---|
| `product_id` | string |
| `product_name`, `brand_name` | string or null |
| `primary_category`, `secondary_category`, `tertiary_category` | string or null |
| `price_usd`, `rating` | finite number or null |
| `out_of_stock` | boolean or null |

There is no image URL, shopping URL, ingredient list or inventory timestamp in this response contract.

### Explanation contracts

`explanation.type` selects exactly one shape:

**Collaborative**

```text
type: "collaborative"
because_you_liked: array of up to 3 objects:
  product_id: string
  product_name: string | null
  rating_weight: finite number
  similarity: finite number
  weighted_contribution: finite number
  score_contribution: finite number
final_score: finite number
profile_weight_sum: finite number
other_score_contribution: finite number
```

`weighted_contribution = rating_weight × similarity`; divide by `profile_weight_sum` for `score_contribution`. The displayed score contributions plus `other_score_contribution` reproduce `final_score`. These are overlapping-user associations, not causal effects.

**Content**

```json
{ "type": "content_fallback", "matching_terms": ["example term"] }
```

The array contains up to five actual matching TF-IDF terms; the term above is illustrative, not a product claim.

**Skin profile**

```text
type: "skin_profile"
skin_type: string | null
skin_tone: string | null
global_train_positive_rate: finite number
signals:
  overall: AffinitySignal
  skin_type: AffinitySignal | null
  skin_tone: AffinitySignal | null
  exact: AffinitySignal | null
final_score: finite number

AffinitySignal:
  smoothed_rate: finite number
  positive_count: integer
  interaction_count: integer
  weight: finite number
```

Missing attributes have null signals. Effective signal weights and smoothed rates combine into the final score. `global_train_positive_rate` is the schema's retained name; for a production bundle, “train” means full processed history. Explicit profiles use the existing learned tables, not newly fitted request data.

**Popularity**

```json
{ "type": "popularity", "positive_user_count": 12 }
```

Counts and terms in these examples are synthetic. The frontend displays preference-oriented language and never medical effectiveness claims.

## GET /products/{product_id}

Returns HTTP 200 with exactly the `ProductMetadata` fields above, without recommendation scores or explanations. URL-encode the product ID in clients. The metadata index includes all 2,420 catalog products, even the 69 cold products that cannot enter rankings.

Unknown product:

```json
{ "detail": "Unknown product" }
```

Status: **404**. A metadata lookup does not add a product to the candidate catalog.

## Errors and empty results

| Condition | Behavior |
|---|---|
| Invalid body, extra field, invalid K, negative/nonfinite price | HTTP 422. `detail` is an array with `loc`, `msg`, `type`; offending input is not echoed. |
| Unknown product | HTTP 404 with the detail above. |
| Missing/corrupt/incompatible bundle | Startup failure; build/rebuild a trusted bundle. |
| Unexpected serving error | Server error; not a normal fallback or empty-result signal. |
| Valid filters with no matches | HTTP 200, `recommendations: []`, `returned_count: 0`, with strategy and filter metadata preserved. |

All schema numeric outputs are finite. Clients should use the status code and safe user messages instead of exposing internal exceptions. No auth, rate limiting, request-history persistence or live catalog synchronization is provided.
