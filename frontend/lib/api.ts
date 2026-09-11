import type { HealthResponse, ProductMetadata, RecommendationRequest, RecommendationResponse } from "./types";

const baseUrl = (process.env.NEXT_PUBLIC_GLOWGUIDE_API_URL || "http://localhost:8000").replace(/\/+$/, "");
export class ApiError extends Error { constructor(message: string, public status?: number) { super(message); this.name = "ApiError"; } }
export const connectionMessage = "GlowGuide couldn't reach the recommendation service. Make sure the API is running and try again.";

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  try {
    const response = await fetch(`${baseUrl}${path}`, { ...options, cache: "no-store", signal: options.signal ?? AbortSignal.timeout(30000) });
    if (!response.ok) {
      const message = response.status === 422 ? "Please check your selections and enter a valid budget, then try again."
        : response.status === 404 ? "This product's details are no longer available."
        : "The recommendation service is temporarily unavailable. Please try again.";
      throw new ApiError(message, response.status);
    }
    return await response.json() as T;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw new ApiError(connectionMessage);
  }
}
export const checkHealth = (signal?: AbortSignal) => request<HealthResponse>("/health", { signal });
export const getRecommendations = (body: RecommendationRequest) => request<RecommendationResponse>("/recommendations", {
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
});
export const getProduct = (id: string, signal?: AbortSignal) => request<ProductMetadata>(`/products/${encodeURIComponent(id)}`, { signal });
