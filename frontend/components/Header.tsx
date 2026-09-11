import { Flower2 } from "lucide-react";
export default function Header() {
  return <header className="site-header"><a className="brand" href="#" aria-label="GlowGuide home"><Flower2 aria-hidden="true" size={29} strokeWidth={1.3} />GlowGuide<span className="brand-dot">.</span></a><span className="header-note">Explainable skincare discovery</span></header>;
}
