// Mirrors src/glowguide/api/schemas.py. No ranking or profile inference in the client.
export type Strategy = "collaborative" | "content_fallback" | "skin_profile" | "popularity";
export interface SkinProfile { skin_type: string | null; skin_tone: string | null }
export interface RecommendationRequest extends SkinProfile {
  user_id: string | null; max_price: number | null; category: string | null; in_stock_only: boolean; top_k: number;
}
export interface ProductMetadata {
  product_id: string; product_name: string | null; brand_name: string | null;
  primary_category: string | null; secondary_category: string | null; tertiary_category: string | null;
  price_usd: number | null; out_of_stock: boolean | null; rating: number | null;
}
export interface HistoryContribution {
  product_id: string; product_name: string | null; rating_weight: number; similarity: number;
  weighted_contribution: number; score_contribution: number;
}
export interface AffinitySignal { smoothed_rate: number; positive_count: number; interaction_count: number; weight: number }
export type Explanation =
  | { type: "collaborative"; because_you_liked: HistoryContribution[]; final_score: number; profile_weight_sum: number; other_score_contribution: number }
  | { type: "content_fallback"; matching_terms: string[] }
  | { type: "skin_profile"; skin_type: string | null; skin_tone: string | null; global_train_positive_rate: number; final_score: number;
      signals: { overall: AffinitySignal; skin_type: AffinitySignal | null; skin_tone: AffinitySignal | null; exact: AffinitySignal | null } }
  | { type: "popularity"; positive_user_count: number };
export interface RecommendationItem extends ProductMetadata {
  score: number;
  score_type: "adjusted_collaborative_affinity" | "tfidf_cosine_similarity" | "smoothed_skin_profile_affinity" | "positive_user_count";
  explanation: Explanation;
}
export interface FiltersApplied { max_price: number | null; price_field: "price_usd"; category: string | null; in_stock_only: boolean }
export interface RecommendationResponse {
  strategy: Strategy; requested_top_k: number; returned_count: number; user_history_available: boolean;
  skin_profile_used: SkinProfile | null; filters_applied: FiltersApplied; candidate_pool_size: number;
  bundle_version: string; recommendations: RecommendationItem[];
}
export interface HealthResponse { status: "ok"; bundle_loaded: boolean; bundle_version: string; candidate_products: number; historical_users: number }
