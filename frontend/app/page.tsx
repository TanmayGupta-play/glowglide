"use client";
import { useEffect, useRef, useState } from "react";
import { ArrowDown, Flower2, Heart, Info } from "lucide-react";
import Header from "@/components/Header";
import PersonalizationForm from "@/components/PersonalizationForm";
import RecommendationGrid from "@/components/RecommendationGrid";
import ExplanationPanel from "@/components/ExplanationPanel";
import ProductDetailPanel from "@/components/ProductDetailPanel";
import StrategyBadge, { strategyCopy } from "@/components/StrategyBadge";
import LoadingState from "@/components/LoadingState";
import EmptyState from "@/components/EmptyState";
import { checkHealth, connectionMessage, getRecommendations } from "@/lib/api";
import { formatUSD, label } from "@/lib/formatting";
import type { RecommendationItem, RecommendationRequest, RecommendationResponse } from "@/lib/types";

export default function Home() {
  const [response, setResponse] = useState<RecommendationResponse | null>(null);
  const [submitted, setSubmitted] = useState<RecommendationRequest | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [unavailable, setUnavailable] = useState(false);
  const [explaining, setExplaining] = useState<RecommendationItem | null>(null);
  const [details, setDetails] = useState<string | null>(null);
  const pending = useRef(false);
  useEffect(() => {
    const controller = new AbortController();
    checkHealth(controller.signal).catch(() => { if (!controller.signal.aborted) setUnavailable(true); });
    return () => controller.abort();
  }, []);
  async function recommend(request: RecommendationRequest) {
    if (pending.current) return;
    pending.current = true; setLoading(true); setError("");
    try { const result = await getRecommendations(request); setResponse(result); setSubmitted(request); setUnavailable(false); }
    catch (error) { setError(error instanceof Error ? error.message : connectionMessage); }
    finally { pending.current = false; setLoading(false); }
  }
  function adjust() { document.getElementById("form-title")?.focus({ preventScroll: true }); document.getElementById("personalization")?.scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth", block: "start" }); }
  const filters = response ? [
    ...(submitted?.skin_type ? [`${label(submitted.skin_type)} skin`] : []),
    ...(submitted?.skin_tone ? [`${label(submitted.skin_tone)} tone`] : []),
    ...(response.filters_applied.max_price !== null ? [`Up to ${formatUSD(response.filters_applied.max_price)}`] : []),
    ...(response.filters_applied.category ? [label(response.filters_applied.category)] : []),
    ...(response.filters_applied.in_stock_only ? ["In stock only"] : []),
  ] : [];
  return <><a href="#personalization" className="skip-link">Skip to personalization</a><div className="page-shell"><Header />
    <main><section className="hero" aria-labelledby="hero-title"><div className="hero-copy"><span className="eyebrow"><span className="small-line" /> Skincare, with you in mind</span><h1 id="hero-title">A little less searching.<br /><em>A little more you.</em></h1><p className="hero-description">Discover skincare using your skin profile and the products you already love.</p><a href="#personalization" className="hero-link">Find your next favorite <ArrowDown size={16} aria-hidden="true" /></a></div><div className="hero-art" aria-hidden="true"><span className="art-caption">THE GLOWGUIDE APPROACH</span><div className="brand-seal"><Flower2 size={43} strokeWidth={.85} /><span>Considered.<br /><em>Personal.</em><br />Always you.</span></div><span className="art-footer">PREFERENCES, MEET POSSIBILITIES</span></div></section>
    <p className="trust-note"><Info size={15} aria-hidden="true" />Recommendations are based on historical product preferences and are not medical advice.</p>
    <div className="discovery-layout"><aside><PersonalizationForm loading={loading} onSubmit={recommend} /></aside><section className="results" aria-labelledby="results-title" aria-busy={loading}>
      <div className="results-top"><div><span className="eyebrow">Thoughtfully discovered</span><h2 id="results-title">Your GlowGuide picks</h2></div>{response && !loading && <span className="result-count">{response.returned_count} {response.returned_count === 1 ? "pick" : "picks"}</span>}</div>
      {unavailable && !error && <p className="service-warning" role="status">The recommendation service is currently unavailable. You can set your preferences and try again shortly.</p>}
      {error && <div className="error-box" role="alert"><strong>Let’s try that again.</strong><p>{error}</p></div>}
      {loading ? <LoadingState /> : response ? <>
        <div className="strategy-row"><StrategyBadge strategy={response.strategy} /><p>{strategyCopy[response.strategy].title}</p></div>
        <div className="chips filter-chips" aria-label="Preferences and applied filters">{filters.map(filter => <span key={filter} className="chip">{filter}</span>)}</div>
        <p className="result-announcement sr-only" role="status">{response.returned_count} recommendations found. {strategyCopy[response.strategy].title}.</p>
        {response.returned_count === 0 ? <EmptyState onAdjust={adjust} /> : <RecommendationGrid products={response.recommendations} onExplain={setExplaining} onDetails={product => setDetails(product.product_id)} />}
      </> : <div className="welcome-state"><div className="welcome-mark"><Flower2 size={50} strokeWidth={1} aria-hidden="true" /></div><span className="eyebrow">Your next discovery starts here</span><h3>Good skincare.<br /><em>Your point of view.</em></h3><p>Tell us a little about your preferences.<br />We’ll find a place for you to start.</p><div className="welcome-points"><span><Heart size={16} aria-hidden="true" />Personal to your preferences</span><span><Info size={16} aria-hidden="true" />A reason behind every pick</span></div><p className="welcome-tip">Just browsing? Leave your profile open to explore community favorites.</p></div>}
      <p className="results-disclosure">GlowGuide recommendations are based on historical preference signals and product metadata. They are for product discovery, not medical advice.</p>
    </section></div></main><footer><span className="footer-brand">GlowGuide.</span><span>A more considered way to discover skincare.</span></footer></div>
    {explaining && <ExplanationPanel product={explaining} onClose={() => setExplaining(null)} />}
    {details && <ProductDetailPanel key={details} productId={details} onClose={() => setDetails(null)} />}
  </>;
}
