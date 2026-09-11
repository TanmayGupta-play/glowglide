import type { Metadata } from "next";
import "./globals.css";
export const metadata: Metadata = { title: "GlowGuide — Skincare, with you in mind", description: "Discover skincare through your preferences, your skin profile, and transparent product recommendations." };
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) { return <html lang="en"><body>{children}</body></html>; }
