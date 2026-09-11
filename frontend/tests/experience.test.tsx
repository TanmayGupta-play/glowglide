import { act, fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import Home from "@/app/page";
import PersonalizationForm from "@/components/PersonalizationForm";
import { checkHealth, getProduct, getRecommendations } from "@/lib/api";
import { product, result } from "./fixtures";
import type { RecommendationResponse, Strategy } from "@/lib/types";
vi.mock("@/lib/api", () => ({ checkHealth: vi.fn(), getRecommendations: vi.fn(), getProduct: vi.fn(), connectionMessage: "GlowGuide couldn't reach the recommendation service. Make sure the API is running and try again." }));
beforeEach(() => { vi.mocked(checkHealth).mockResolvedValue({ status: "ok", bundle_loaded: true, bundle_version: "1", candidate_products: 10, historical_users: 20 }); vi.mocked(getRecommendations).mockReset(); vi.mocked(getProduct).mockReset(); });
const submit = () => userEvent.click(screen.getByRole("button", { name: "Find my skincare" }));

describe("personalization", () => {
  it("renders the form, supports skin selection and deselection", async () => {
    render(<PersonalizationForm loading={false} onSubmit={vi.fn()} />);
    expect(screen.getByLabelText("Skin tone")).toBeInTheDocument();
    const combination = screen.getByRole("button", { name: "Combination" });
    await userEvent.click(combination); expect(combination).toHaveAttribute("aria-pressed", "true");
    await userEvent.click(combination); expect(combination).toHaveAttribute("aria-pressed", "false");
  });
  it("sends exact values and does not require an ID", async () => {
    const handler = vi.fn(); render(<PersonalizationForm loading={false} onSubmit={handler} />);
    await userEvent.click(screen.getByRole("button", { name: "Combination" }));
    await userEvent.selectOptions(screen.getByLabelText("Skin tone"), "medium");
    await userEvent.selectOptions(screen.getByLabelText("Category"), "Moisturizers");
    await userEvent.type(screen.getByLabelText("Maximum price (USD)"), "40");
    await userEvent.selectOptions(screen.getByLabelText("Show me"), "20");
    await userEvent.click(screen.getByLabelText("In stock only")); await submit();
    expect(handler).toHaveBeenCalledWith({ user_id: null, skin_type: "combination", skin_tone: "medium", category: "Moisturizers", max_price: 40, in_stock_only: false, top_k: 20 });
  });
  it("prevents negative budgets even when native validation is bypassed", () => {
    const handler = vi.fn(); render(<PersonalizationForm loading={false} onSubmit={handler} />);
    fireEvent.change(screen.getByLabelText("Maximum price (USD)"), { target: { value: "-1" } });
    fireEvent.submit(screen.getByRole("form")); expect(handler).not.toHaveBeenCalled();
    expect(screen.getByRole("alert")).toHaveTextContent("$0 or more");
  });
});

describe("recommendation experience", () => {
  it("shows skeleton loading and prevents duplicate submissions", async () => {
    let finish!: (value: RecommendationResponse) => void;
    vi.mocked(getRecommendations).mockImplementation(() => new Promise(resolve => { finish = resolve; }));
    render(<Home />); await submit();
    expect(screen.getByText("Finding your matches...")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Finding your matches…" })).toBeDisabled();
    fireEvent.submit(screen.getByRole("form")); expect(getRecommendations).toHaveBeenCalledTimes(1);
    await act(async () => finish(result()));
    expect(screen.getByText("Example Gentle Serum")).toBeInTheDocument();
  });
  it.each<[Strategy, string]>([["collaborative", "Your product history"], ["skin_profile", "Your skin profile"], ["popularity", "Community favorites"], ["content_fallback", "Your preferences"]])("renders %s with a human-readable strategy", async (strategy, badge) => {
    vi.mocked(getRecommendations).mockResolvedValue(result(strategy)); render(<Home />); await submit();
    expect(await screen.findByText("Example Gentle Serum")).toBeInTheDocument(); expect(screen.getByText(badge)).toBeInTheDocument();
    expect(screen.getByText("$24.00")).toBeInTheDocument();
  });
  it.each<[Strategy, string]>([["collaborative", "Example Aloe Cream"], ["content_fallback", "aloe"], ["skin_profile", "Community overall"], ["popularity", "321"]])("shows actual %s explanation without fetching and preserves results", async (strategy, expected) => {
    vi.mocked(getRecommendations).mockResolvedValue(result(strategy)); render(<Home />); await submit();
    const why = await screen.findByRole("button", { name: "Why this?" }); await userEvent.click(why);
    expect(screen.getByRole("dialog")).toHaveAccessibleName("Why this made your list");
    expect(within(screen.getByRole("dialog")).getByText(expected)).toBeInTheDocument();
    expect(getProduct).not.toHaveBeenCalled();
    await userEvent.keyboard("{Escape}"); expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(why).toHaveFocus(); expect(screen.getByText("Example Gentle Serum")).toBeInTheDocument(); expect(getRecommendations).toHaveBeenCalledTimes(1);
  });
  it("shows an empty state and returns focus to filters", async () => {
    vi.mocked(getRecommendations).mockResolvedValue({ ...result(), returned_count: 0, recommendations: [] }); render(<Home />); await submit();
    expect(await screen.findByText("No products matched all of those filters.")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Adjust filters" })); expect(screen.getByRole("heading", { name: "Find your kind of skincare." })).toHaveFocus();
  });
  it("shows a human-friendly backend failure", async () => {
    vi.mocked(getRecommendations).mockRejectedValue(new Error("GlowGuide couldn't reach the recommendation service. Make sure the API is running and try again.")); render(<Home />); await submit();
    expect(await screen.findByRole("alert")).toHaveTextContent("Make sure the API is running");
  });
  it("preserves backend order and handles missing optional metadata", async () => {
    const a = { ...product(), product_name: null, brand_name: null, price_usd: null, secondary_category: null, tertiary_category: null };
    const b = { ...product(), product_id: "example-b", product_name: "Second product" };
    vi.mocked(getRecommendations).mockResolvedValue({ ...result(), returned_count: 2, recommendations: [a, b] }); render(<Home />); await submit();
    const cards = await screen.findAllByRole("article"); expect(cards).toHaveLength(2);
    expect(cards[0]).toHaveTextContent("Skincare product"); expect(cards[1]).toHaveTextContent("Second product");
    expect(screen.getByText("Price unavailable")).toBeInTheDocument(); expect(document.body.textContent).not.toMatch(/undefined|NaN|Infinity/);
  });
  it("fetches product details only on explicit action and closes accessibly", async () => {
    vi.mocked(getRecommendations).mockResolvedValue(result());
    const metadata = product();
    vi.mocked(getProduct).mockResolvedValue(metadata); render(<Home />); await submit();
    await userEvent.click(await screen.findByRole("button", { name: /View details for/ }));
    expect(getProduct).toHaveBeenCalledWith("example-a", expect.any(AbortSignal));
    expect(await screen.findByText("4.2 / 5")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Close dialog" })); expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });
  it("keeps filter summary tied to the submitted request", async () => {
    vi.mocked(getRecommendations).mockResolvedValue(result("skin_profile")); render(<Home />);
    await userEvent.click(screen.getByRole("button", { name: "Combination" })); await submit();
    await screen.findByText("Combination skin"); await userEvent.click(screen.getByRole("button", { name: "Dry" }));
    expect(screen.getByText("Combination skin")).toBeInTheDocument(); expect(screen.queryByText("Dry skin")).not.toBeInTheDocument();
  });
});
