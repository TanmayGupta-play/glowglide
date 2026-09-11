import { ArrowUpRight, Plus } from "lucide-react";
import { categoryLabel, displayText, formatUSD } from "@/lib/formatting";
import type { RecommendationItem } from "@/lib/types";
export default function ProductCard({ product, index, onExplain, onDetails }: { product: RecommendationItem; index: number; onExplain: () => void; onDetails: () => void }) {
  const initials = displayText(product.brand_name, "GG").split(/\s+/).slice(0, 2).map(word => word[0]).join("");
  return <article className="product-card"><div className={`product-surface surface-${index % 4}`} aria-hidden="true"><span className="product-number">{String(index + 1).padStart(2, "0")}</span><span className="monogram">{initials}</span><span className="surface-category">{categoryLabel(product)}</span><span className="surface-circle" /></div>
    <div className="product-body"><p className="product-brand">{displayText(product.brand_name, "Skincare discovery")}</p><h3>{displayText(product.product_name, "Skincare product")}</h3><p className="product-category">{categoryLabel(product)}</p><p className="product-price">{formatUSD(product.price_usd)}</p>
    <div className="product-actions"><button onClick={onExplain} className="why-button">Why this? <Plus size={15} aria-hidden="true" /></button><button onClick={onDetails} className="details-button" aria-label={`View details for ${displayText(product.product_name, "Skincare product")}`}>View details <ArrowUpRight size={14} aria-hidden="true" /></button></div></div></article>;
}
