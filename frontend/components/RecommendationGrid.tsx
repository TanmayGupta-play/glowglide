import ProductCard from "./ProductCard";
import type { RecommendationItem } from "@/lib/types";
export default function RecommendationGrid({ products, onExplain, onDetails }: { products: RecommendationItem[]; onExplain: (product: RecommendationItem) => void; onDetails: (product: RecommendationItem) => void }) {
  return <div className="recommendation-grid">{products.map((product, index) => <ProductCard key={product.product_id} product={product} index={index} onExplain={() => onExplain(product)} onDetails={() => onDetails(product)} />)}</div>;
}
