"use client";
import { useEffect, useState } from "react";
import { getProduct } from "@/lib/api";
import { categoryLabel, displayText, formatUSD } from "@/lib/formatting";
import type { ProductMetadata } from "@/lib/types";
import Modal from "./Modal";
export default function ProductDetailPanel({ productId, onClose }: { productId: string; onClose: () => void }) {
  const [product, setProduct] = useState<ProductMetadata | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    getProduct(productId, controller.signal).then(setProduct).catch(error => { if (!controller.signal.aborted) setError(error instanceof Error ? error.message : "Product details couldn't be loaded."); });
    return () => controller.abort();
  }, [productId]);
  return <Modal title={product ? displayText(product.product_name, "Product details") : "Product details"} onClose={onClose}>
    {error ? <p role="alert" className="error-box">{error}</p> : !product ? <p role="status">Loading product details…</p> : <><p className="product-brand">{displayText(product.brand_name, "Brand unavailable")}</p><p className="detail-price">{formatUSD(product.price_usd)}</p><dl className="affinity-list"><div><dt>Category</dt><dd>{categoryLabel(product)}</dd></div><div><dt>Catalog rating</dt><dd>{product.rating !== null && Number.isFinite(product.rating) ? `${product.rating.toFixed(1)} / 5` : "Not available"}</dd></div><div><dt>Availability</dt><dd>{product.out_of_stock === true ? "Out of stock" : product.out_of_stock === false ? "In stock" : "Stock status unavailable"}</dd></div></dl><p className="field-hint">Prices are in USD, as listed in the product catalog. Availability reflects the catalog data.</p></>}
  </Modal>;
}
