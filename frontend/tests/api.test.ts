import { afterEach, expect, it, vi } from "vitest";
import { checkHealth, connectionMessage, getProduct, getRecommendations } from "@/lib/api";
import { formatUSD } from "@/lib/formatting";
afterEach(() => vi.unstubAllGlobals());
it("uses the API URL, encodes product IDs, and handles non-2xx", async () => {
  const fetcher = vi.fn().mockResolvedValue(new Response("{}", { status: 404 })); vi.stubGlobal("fetch", fetcher);
  await expect(getProduct("a/b")).rejects.toThrow("no longer available");
  expect(fetcher.mock.calls[0][0]).toBe("http://localhost:8000/products/a%2Fb");
});
it("posts JSON and translates validation failures", async () => {
  const fetcher = vi.fn().mockResolvedValue(new Response("{}", { status: 422 })); vi.stubGlobal("fetch", fetcher);
  const body = { user_id: null, skin_type: null, skin_tone: null, max_price: 1, category: null, top_k: 10, in_stock_only: true };
  await expect(getRecommendations(body)).rejects.toThrow("check your selections");
  expect(fetcher.mock.calls[0][1].body).toBe(JSON.stringify(body));
});
it("translates connection errors without exposing stack traces", async () => {
  vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("fetch failed")));
  await expect(checkHealth()).rejects.toThrow(connectionMessage);
});
it("never formats missing or nonfinite prices as numbers", () => {
  for (const price of [null, undefined, NaN, Infinity]) expect(formatUSD(price)).toBe("Price unavailable");
  expect(formatUSD(0)).toBe("$0.00");
});
