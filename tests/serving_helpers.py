"""Small shared serving/API fixture; no real project data or artifacts."""

import pandas as pd

from glowguide.serving_artifacts import build_serving_bundle


def small_bundle():
    interactions = pd.DataFrame([
        ("u", "A", 1, 4), ("u", "N", 0, 1),
        ("v", "A", 1, 5), ("v", "B", 1, 4), ("v", "C", 1, 5),
        ("v", "E", 1, 4), ("v", "F", 1, 4), ("v", "N", 1, 4),
        ("solo", "D", 1, 4), ("negative", "N", 0, 1),
    ], columns=["author_id", "product_id", "positive", "rating"]).assign(
        submission_time="2023-01-01", skin_type="combination", skin_tone="light_medium")
    products = pd.DataFrame([
        ("A", "Aloe serum", "Serums", 10, 0), ("B", "Aloe hydrating serum", "Serums", 20, "false"),
        ("C", "Gentle cream", "Moisturizers", 40, "1"), ("D", "Aloe gentle cream", "Moisturizers", 30, False),
        ("E", "Hydrating cream", "Moisturizers", None, None), ("F", "Aloe face serum", " Face   Serums ", 5, "0"),
        ("N", "Aloe lotion", "Serums", 15, False), ("COLD", "Cold catalog cream", "Moisturizers", 99, True),
    ], columns=["product_id", "product_name", "secondary_category", "price_usd", "out_of_stock"]).assign(
        brand_name="Example", primary_category="Skincare", rating=float("nan"))
    return build_serving_bundle(interactions, products)
