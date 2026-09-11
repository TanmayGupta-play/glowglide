import { Heart } from "lucide-react";
import type { Strategy } from "@/lib/types";
export const strategyCopy: Record<Strategy, { badge: string; title: string; description: string }> = {
  collaborative: { badge: "Your product history", title: "Based on products you've liked", description: "Products connected by positive historical interactions with the products you liked." },
  content_fallback: { badge: "Your preferences", title: "Based on your product preferences", description: "Shared product metadata connects these picks with your positive product history." },
  skin_profile: { badge: "Your skin profile", title: "Based on similar skin profiles", description: "Historical product affinity for your supplied skin type and tone." },
  popularity: { badge: "Community favorites", title: "Popular skincare picks", description: "Products liked by distinct users in historical interactions." },
};
export default function StrategyBadge({ strategy }: { strategy: Strategy }) { return <span className="strategy-badge" title={strategyCopy[strategy].description}><Heart size={14} aria-hidden="true" />{strategyCopy[strategy].badge}</span>; }
