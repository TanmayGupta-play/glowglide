import Modal from "./Modal";
import { displayText, label, numberLabel } from "@/lib/formatting";
import type { RecommendationItem } from "@/lib/types";

export default function ExplanationPanel({ product, onClose }: { product: RecommendationItem; onClose: () => void }) {
  const explanation = product.explanation;
  return <Modal title="Why this made your list" onClose={onClose}><p className="modal-product">{displayText(product.product_name, "This skincare product")}</p>
    {explanation.type === "collaborative" && <><h3>Recommended from your product history</h3><p>People who liked products in your history also positively interacted with this product. These connections contributed to your recommendation.</p><ul className="history-products">{explanation.because_you_liked.map(item => <li key={item.product_id}><span className="tiny-mark" aria-hidden="true">G</span>{displayText(item.product_name, "A product in your history")}</li>)}</ul></>}
    {explanation.type === "content_fallback" && <><h3>Matches themes in products you’ve liked</h3><p>These terms are shared with your product preferences, using the catalog’s product descriptions and metadata.</p><div className="chips">{explanation.matching_terms.map(term => <span key={term} className="chip">{term}</span>)}</div>{!explanation.matching_terms.length && <p>No shared terms are available to display.</p>}</>}
    {explanation.type === "skin_profile" && <><h3>Based on historical preferences from similar skin profiles</h3><div className="chips">{explanation.skin_type && <span className="chip">{label(explanation.skin_type)} skin</span>}{explanation.skin_tone && <span className="chip">{label(explanation.skin_tone)} tone</span>}</div><p>Learned affinity reflects historical preferences, not medical effectiveness.</p><dl className="affinity-list">{([
      ["overall", "Community overall"], ["skin_type", "Skin type"], ["skin_tone", "Skin tone"], ["exact", "Type & tone together"],
    ] as const).map(([key, title]) => { const signal = explanation.signals[key]; return signal ? <div key={key}><dt>{title}</dt><dd><strong>{Number.isFinite(signal.smoothed_rate) ? signal.smoothed_rate.toFixed(3) : "Not available"}</strong> affinity<span>{numberLabel(signal.positive_count)} positive / {numberLabel(signal.interaction_count)} historical interactions</span></dd></div> : null; })}</dl><p className="field-hint">Affinity values are smoothed estimates. Groups with little history use broader historical preferences.</p></>}
    {explanation.type === "popularity" && <><h3>Frequently liked by the community</h3><p className="community-count">{numberLabel(explanation.positive_user_count)}</p><p>Distinct users positively interacted with this product in the historical data.</p></>}
    <p className="modal-disclosure">A starting point for discovery, not a promise of results or medical advice.</p>
  </Modal>;
}
