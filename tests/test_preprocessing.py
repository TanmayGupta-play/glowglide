"""Small synthetic tests; no access to the real skincare dataset is needed."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from glowguide.data import load_products, load_reviews
from glowguide.preprocessing import (
    build_interactions,
    clean_products,
    clean_reviews,
    deduplicate_interactions,
    normalize_profile_value,
)


def review_frame(**overrides: object) -> pd.DataFrame:
    row = {
        "author_id": "001", "product_id": "p1", "rating": 5,
        "is_recommended": 1, "submission_time": "2024-01-01",
        "skin_type": " Dry ", "skin_tone": "lightMedium",
        "eye_color": None, "hair_color": None,
    }
    row.update(overrides)
    return pd.DataFrame([row])


class PreprocessingTests(unittest.TestCase):
    def test_preference_labels(self):
        for rating, recommended, positive, strong in [
            (5, 1, 1, 1), (4, 1, 1, 0), (3, 1, 0, 0),
            (1, 0, 0, 0), (5, 0, 1, 0), (5, None, 1, 0),
            (4, None, 1, 0),
        ]:
            with self.subTest(rating=rating, recommended=recommended):
                result = clean_reviews(review_frame(rating=rating, is_recommended=recommended))
                self.assertEqual(len(result), 1)
                self.assertEqual(result.loc[0, "positive"], positive)
                self.assertEqual(result.loc[0, "strong_positive"], strong)
                self.assertEqual(str(result["positive"].dtype), "int8")
                self.assertEqual(str(result["strong_positive"].dtype), "int8")
                self.assertEqual(str(result["is_recommended"].dtype), "Float64")
                if recommended is None:
                    self.assertTrue(pd.isna(result.loc[0, "is_recommended"]))

    def test_latest_duplicate_wins_even_when_negative(self):
        reviews = pd.concat([
            review_frame(submission_time="2024-02-01", rating=2),
            review_frame(submission_time="2024-01-01", rating=5),
            review_frame(author_id="002"),
            review_frame(product_id="p2"),
        ], ignore_index=True)
        result = deduplicate_interactions(clean_reviews(reviews))
        self.assertEqual(len(result), 3)
        latest = result.loc[result["author_id"].eq("001") & result["product_id"].eq("p1")].iloc[0]
        self.assertEqual(latest["submission_time"], pd.Timestamp("2024-02-01"))
        self.assertEqual(latest["positive"], 0)

    def test_timestamp_ties_keep_last_source_row(self):
        reviews = pd.concat([review_frame(rating=5), review_frame(rating=2)], ignore_index=True)
        result = deduplicate_interactions(clean_reviews(reviews))
        self.assertEqual(result["rating"].tolist(), [2])

    def test_catalog_filter_and_product_cleaning(self):
        products = pd.DataFrame({
            "product_id": [" p1 ", "p1", "p2", "p3", None, " "],
            "primary_category": [" sKiNcArE ", "Skincare", "Makeup", "Skincare", "Skincare", "Skincare"],
            "product_name": ["First", "Duplicate", "Makeup", "Unreviewed", "Absent", "Blank"],
            "ingredients": [None] * 6,
        })
        cleaned = clean_products(products)
        self.assertEqual(cleaned["product_id"].tolist(), ["p1", "p3"])
        self.assertEqual(cleaned.loc[0, "product_name"], "First")
        self.assertTrue(cleaned["ingredients"].isna().all())
        self.assertTrue(cleaned["highlights"].isna().all())
        reviews = pd.concat([
            review_frame(), review_frame(product_id="p2"), review_frame(product_id="unknown"),
        ], ignore_index=True)
        result = build_interactions(products, reviews)
        self.assertEqual(result["product_id"].tolist(), ["p1"])

    def test_ids_are_nullable_strings_and_trimmed(self):
        result = clean_reviews(review_frame(author_id=" 001 ", product_id=" 002 "))
        self.assertEqual(result.loc[0, "author_id"], "001")
        self.assertEqual(result.loc[0, "product_id"], "002")
        for column in ("author_id", "product_id"):
            self.assertIsInstance(result[column].dtype, pd.StringDtype)
        numeric = clean_reviews(review_frame(author_id=123, product_id=456))
        self.assertEqual(numeric.loc[0, "author_id"], "123")
        products = clean_products(pd.DataFrame({"product_id": [456], "primary_category": ["Skincare"]}))
        self.assertIsInstance(products["product_id"].dtype, pd.StringDtype)
        self.assertEqual(products.loc[0, "product_id"], "456")

    def test_profile_normalization(self):
        for value, expected in [
            ("lightMedium", "light_medium"), ("mediumTan", "medium_tan"),
            ("fairLight", "fair_light"), (" DRY ", "dry"),
            ("Light Medium", "light_medium"), ("medium-tan", "medium_tan"),
        ]:
            with self.subTest(value=value):
                self.assertEqual(normalize_profile_value(value), expected)
        for missing in (None, pd.NA, float("nan"), " "):
            self.assertTrue(pd.isna(normalize_profile_value(missing)))
        result = clean_reviews(review_frame())
        self.assertEqual(result.loc[0, "skin_tone"], "light_medium")
        self.assertEqual(result.loc[0, "skin_type"], "dry")
        self.assertTrue(pd.isna(result.loc[0, "eye_color"]))

    def test_all_optional_fields_can_be_absent(self):
        reviews = review_frame().drop(columns=[
            "is_recommended", "skin_type", "skin_tone", "eye_color", "hair_color",
        ])
        result = clean_reviews(reviews)
        self.assertEqual(len(result), 1)
        self.assertTrue(result[["skin_type", "skin_tone", "eye_color", "hair_color"]].isna().all().all())
        self.assertTrue(pd.isna(result.loc[0, "is_recommended"]))
        self.assertEqual(result.loc[0, "strong_positive"], 0)

    def test_invalid_required_values_are_removed(self):
        for overrides in [
            {"author_id": None}, {"author_id": " "}, {"product_id": None},
            {"product_id": " "}, {"rating": "invalid"}, {"rating": None},
            {"rating": 0}, {"rating": 6},
            {"rating": float("inf")}, {"submission_time": "invalid"},
            {"submission_time": None},
        ]:
            with self.subTest(overrides=overrides):
                self.assertTrue(clean_reviews(review_frame(**overrides)).empty)
        result = clean_reviews(review_frame(rating="4", is_recommended="unknown"))
        self.assertEqual(len(result), 1)
        self.assertEqual(result.loc[0, "positive"], 1)
        self.assertTrue(pd.isna(result.loc[0, "is_recommended"]))

    def test_missing_required_columns_raise_clear_errors(self):
        with self.assertRaisesRegex(ValueError, "Reviews.*author_id"):
            clean_reviews(review_frame().drop(columns="author_id"))
        with self.assertRaisesRegex(ValueError, "Products.*primary_category"):
            clean_products(pd.DataFrame({"product_id": ["p1"]}))

    def test_inputs_are_not_mutated_and_empty_input_is_supported(self):
        products = pd.DataFrame({"product_id": ["p1"], "primary_category": ["Skincare"]})
        reviews = review_frame()
        products_before, reviews_before = products.copy(deep=True), reviews.copy(deep=True)
        build_interactions(products, reviews)
        assert_frame_equal(products, products_before)
        assert_frame_equal(reviews, reviews_before)
        self.assertTrue(build_interactions(products, reviews.iloc[:0]).empty)


class LoaderTests(unittest.TestCase):
    def test_discovery_order_columns_and_ids(self):
        with TemporaryDirectory() as directory:
            raw_dir = Path(directory)
            review_frame(author_id="0002", product_id="002", rating=2).to_csv(raw_dir / "reviews_b.csv")
            review_frame(author_id="0001", product_id="002", rating=5).to_csv(raw_dir / "reviews_a.csv")
            review_frame().to_csv(raw_dir / "unrelated.csv")
            reviews = load_reviews(raw_dir)
            self.assertEqual(reviews["author_id"].tolist(), ["0001", "0002"])
            self.assertEqual(reviews["product_id"].tolist(), ["002", "002"])
            self.assertNotIn("Unnamed: 0", reviews.columns)
            result = deduplicate_interactions(clean_reviews(reviews.assign(author_id="same")))
            self.assertEqual(result["rating"].tolist(), [2])
            pd.DataFrame({
                "product_id": ["002"], "primary_category": ["Skincare"], "unused": [1],
            }).to_csv(raw_dir / "product_info.csv")
            products = load_products(raw_dir)
            self.assertEqual(products["product_id"].tolist(), ["002"])
            self.assertNotIn("unused", products.columns)
            self.assertNotIn("Unnamed: 0", products.columns)

    def test_missing_files_and_columns_raise_clear_errors(self):
        with TemporaryDirectory() as directory:
            raw_dir = Path(directory)
            with self.assertRaisesRegex(FileNotFoundError, "product_info.csv"):
                load_products(raw_dir)
            with self.assertRaisesRegex(FileNotFoundError, "reviews_.*csv"):
                load_reviews(raw_dir)
            pd.DataFrame({"rating": [5]}).to_csv(raw_dir / "reviews_bad.csv", index=False)
            with self.assertRaisesRegex(ValueError, "reviews_bad.csv.*author_id"):
                load_reviews(raw_dir)


if __name__ == "__main__":
    unittest.main()
