"use client";
import { useState, useSyncExternalStore, type FormEvent } from "react";
import { ArrowUpRight, Check, SlidersHorizontal } from "lucide-react";
import { categories, skinTones, skinTypes } from "@/lib/options";
import { label } from "@/lib/formatting";
import type { RecommendationRequest } from "@/lib/types";

const subscribe = () => () => {};
export default function PersonalizationForm({ loading, onSubmit }: { loading: boolean; onSubmit: (value: RecommendationRequest) => void }) {
  // Keep the SSR form inert until client handlers attach; never submit IDs in a URL.
  const ready = useSyncExternalStore(subscribe, () => true, () => false);
  const [skinType, setSkinType] = useState<string | null>(null);
  const [budgetError, setBudgetError] = useState("");
  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (loading) return;
    const data = new FormData(event.currentTarget);
    const text = (key: string) => String(data.get(key) ?? "").trim() || null;
    const budget = text("max_price");
    const maxPrice = budget === null ? null : Number(budget);
    if (maxPrice !== null && (!Number.isFinite(maxPrice) || maxPrice < 0)) { setBudgetError("Enter a price of $0 or more, or leave it blank."); return; }
    setBudgetError("");
    onSubmit({ user_id: text("user_id"), skin_type: skinType, skin_tone: text("skin_tone"), max_price: maxPrice,
      category: text("category"), in_stock_only: data.get("in_stock_only") === "on", top_k: Number(text("top_k") || 10) });
  }
  return <form id="personalization" className="personalization" onSubmit={submit} aria-labelledby="form-title">
    <div className="form-heading"><span className="eyebrow">A little about you</span><SlidersHorizontal size={19} aria-hidden="true" /></div>
    <h2 id="form-title" tabIndex={-1}>Find your kind of skincare.</h2><p className="muted form-intro">Start with your skin profile, or explore community favorites. Every field is optional.</p>
    <fieldset disabled={loading || !ready} className="form-fields"><legend className="sr-only">Personalization and filters</legend>
      <fieldset className="skin-type-group"><legend>What’s your skin type?</legend><div className="skin-options">
        {skinTypes.map(type => <button key={type} type="button" className={`skin-option ${skinType === type ? "selected" : ""}`} aria-pressed={skinType === type} onClick={() => setSkinType(skinType === type ? null : type)}>{label(type)}{skinType === type ? <Check size={16} aria-hidden="true" /> : <span className="selection-circle" aria-hidden="true" />}</button>)}
      </div><span className="field-hint">Not sure? You can leave this open.</span></fieldset>
      <label className="field">Skin tone<select name="skin_tone" defaultValue=""><option value="">No preference</option>{skinTones.map(tone => <option key={tone} value={tone}>{label(tone)}</option>)}</select></label>
      <div className="form-divider" /><span className="eyebrow">Make it yours</span>
      <label className="field">Category<select name="category" defaultValue=""><option value="">All skincare</option>{categories.map(category => <option key={category}>{category}</option>)}</select></label>
      <div className="field-pair"><label className="field">Maximum price (USD)<input type="number" name="max_price" min="0" step="0.01" placeholder="No limit" aria-describedby={budgetError ? "budget-error" : undefined} aria-invalid={!!budgetError} onChange={() => setBudgetError("")} /></label>
      <label className="field">Show me<select name="top_k" defaultValue="10">{[5, 10, 20].map(k => <option key={k} value={k}>{k} products</option>)}</select></label></div>
      {budgetError && <p id="budget-error" role="alert" className="field-error">{budgetError}</p>}
      <label className="toggle-label"><span>In stock only</span><input name="in_stock_only" type="checkbox" defaultChecked className="toggle" /></label>
      <div className="history-field"><label className="field" htmlFor="user-id">Already explored products with us?<input id="user-id" name="user_id" placeholder="Optional demo user ID" autoComplete="off" aria-describedby="history-help" /></label><p id="history-help" className="field-hint">Enter a known demo user ID to personalize from product history.</p></div>
      <button type="submit" className="primary-button">{loading ? "Finding your matches…" : "Find my skincare"}<ArrowUpRight size={19} aria-hidden="true" /></button>
    </fieldset><p className="form-footnote">Your preferences. A more thoughtful starting point.</p>
  </form>;
}
