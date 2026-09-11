"""Explicit request/response contracts; scores are affinities, not probabilities."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, field_validator

from ..serving import normalized_profile


class APIModel(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")


class RecommendationRequest(APIModel):
    user_id: str | None = None
    skin_type: str | None = None
    skin_tone: str | None = None
    max_price: Annotated[FiniteFloat, Field(ge=0)] | None = None
    category: str | None = None
    in_stock_only: bool = True
    top_k: Annotated[int, Field(ge=1, le=50, strict=True)] = 10

    @field_validator("user_id", "skin_type", "skin_tone", "category", mode="before")
    @classmethod
    def trim_empty(cls, value):
        return value.strip() or None if isinstance(value, str) else value

    @field_validator("skin_type", "skin_tone")
    @classmethod
    def normalize_profile(cls, value):
        return normalized_profile(value)


class SkinProfile(APIModel):
    skin_type: str | None = None
    skin_tone: str | None = None


class ProductMetadata(APIModel):
    product_id: str
    product_name: str | None = None
    brand_name: str | None = None
    primary_category: str | None = None
    secondary_category: str | None = None
    tertiary_category: str | None = None
    price_usd: FiniteFloat | None = None
    out_of_stock: bool | None = None
    rating: FiniteFloat | None = None


class HistoryContribution(APIModel):
    product_id: str
    product_name: str | None = None
    rating_weight: FiniteFloat
    similarity: FiniteFloat
    weighted_contribution: FiniteFloat
    score_contribution: FiniteFloat


class CollaborativeExplanation(APIModel):
    type: Literal["collaborative"]
    because_you_liked: list[HistoryContribution]
    final_score: FiniteFloat
    profile_weight_sum: FiniteFloat
    other_score_contribution: FiniteFloat


class ContentExplanation(APIModel):
    type: Literal["content_fallback"]
    matching_terms: list[str]


class AffinitySignal(APIModel):
    smoothed_rate: FiniteFloat
    positive_count: int
    interaction_count: int
    weight: FiniteFloat


class AffinitySignals(APIModel):
    overall: AffinitySignal
    skin_type: AffinitySignal | None
    skin_tone: AffinitySignal | None
    exact: AffinitySignal | None


class ProfileExplanation(SkinProfile):
    type: Literal["skin_profile"]
    global_train_positive_rate: FiniteFloat
    signals: AffinitySignals
    final_score: FiniteFloat


class PopularityExplanation(APIModel):
    type: Literal["popularity"]
    positive_user_count: int


Explanation = Annotated[CollaborativeExplanation | ContentExplanation | ProfileExplanation | PopularityExplanation, Field(discriminator="type")]


class RecommendationItem(ProductMetadata):
    score: FiniteFloat
    score_type: Literal["adjusted_collaborative_affinity", "tfidf_cosine_similarity", "smoothed_skin_profile_affinity", "positive_user_count"]
    explanation: Explanation


class FiltersApplied(APIModel):
    max_price: FiniteFloat | None
    price_field: Literal["price_usd"]
    category: str | None
    in_stock_only: bool


class RecommendationResponse(APIModel):
    strategy: Literal["collaborative", "content_fallback", "skin_profile", "popularity"]
    requested_top_k: int
    returned_count: int
    user_history_available: bool
    skin_profile_used: SkinProfile | None
    filters_applied: FiltersApplied
    candidate_pool_size: int
    bundle_version: str
    recommendations: list[RecommendationItem]


class HealthResponse(APIModel):
    status: Literal["ok"]
    bundle_loaded: bool
    bundle_version: str
    candidate_products: int
    historical_users: int
