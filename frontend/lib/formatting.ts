import type { ProductMetadata } from "./types";
export const displayText = (value: string | null | undefined, fallback = "Not available") => value?.trim() || fallback;
export const formatUSD = (value: number | null | undefined) => typeof value === "number" && Number.isFinite(value)
  ? new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 2 }).format(value) : "Price unavailable";
export const label = (value: string | null | undefined) => value ? value === "not_sure_st" ? "Not sure" : value.replaceAll("_", " ").replace(/\b\w/g, c => c.toUpperCase()) : "Not selected";
export const categoryLabel = (product: ProductMetadata) => product.tertiary_category || product.secondary_category || product.primary_category || "Skincare";
export const numberLabel = (value: number) => Number.isFinite(value) ? new Intl.NumberFormat("en-US").format(value) : "Not available";
