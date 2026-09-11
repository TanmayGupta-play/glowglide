import { SlidersHorizontal } from "lucide-react";
export default function EmptyState({ onAdjust }: { onAdjust: () => void }) {
  return <div className="empty-state"><span className="empty-icon"><SlidersHorizontal size={26} aria-hidden="true" /></span><h3>No products matched all of those filters.</h3><p>A little more room could help. Try increasing your budget, removing the category filter, or allowing out-of-stock products.</p><button className="secondary-button" onClick={onAdjust}>Adjust filters</button></div>;
}
