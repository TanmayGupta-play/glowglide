from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ================================================================
# 1. PRODUCT DATA
# ================================================================

print_section("PRODUCT DATA")

product_path = RAW_DIR / "product_info.csv"

products = pd.read_csv(
    product_path,
    low_memory=False
)

print("Total products:", len(products))
print("Unique product IDs:", products["product_id"].nunique())


print("\nPrimary category distribution:")
print(
    products["primary_category"]
    .value_counts(dropna=False)
)


# ----------------------------------------------------------------
# Skincare products
# ----------------------------------------------------------------

skincare_products = products[
    products["primary_category"]
    .astype(str)
    .str.lower()
    .eq("skincare")
].copy()

print("\nTotal skincare products:", len(skincare_products))


print("\nSkincare secondary categories:")
print(
    skincare_products["secondary_category"]
    .value_counts(dropna=False)
    .head(20)
)


print("\nSkincare tertiary categories:")
print(
    skincare_products["tertiary_category"]
    .value_counts(dropna=False)
    .head(30)
)


# ----------------------------------------------------------------
# Important product fields
# ----------------------------------------------------------------

print_section("PRODUCT MISSING VALUES")

product_columns = [
    "product_id",
    "product_name",
    "brand_name",
    "loves_count",
    "rating",
    "reviews",
    "ingredients",
    "price_usd",
    "highlights",
    "primary_category",
    "secondary_category",
    "tertiary_category",
    "out_of_stock",
]


for column in product_columns:

    missing = products[column].isna().sum()

    percentage = (
        missing / len(products)
    ) * 100

    print(
        f"{column:25}"
        f"{missing:8} missing "
        f"({percentage:.2f}%)"
    )


# ================================================================
# 2. REVIEW DATA
# ================================================================

print_section("REVIEW DATA")

review_files = sorted(
    RAW_DIR.glob("reviews_*.csv")
)

print("Review files:", len(review_files))


review_columns = [
    "author_id",
    "rating",
    "is_recommended",
    "submission_time",
    "skin_tone",
    "skin_type",
    "eye_color",
    "hair_color",
    "product_id",
]


total_reviews = 0

unique_users = set()
unique_products = set()

# Important:
# This lets us distinguish review rows from actual unique
# user-product interactions.
unique_user_product_pairs = set()


rating_counter = Counter()
recommended_counter = Counter()

skin_type_counter = Counter()
skin_tone_counter = Counter()

user_interaction_counter = Counter()
product_interaction_counter = Counter()

missing_counter = Counter()


positive_rating_only = 0
positive_rating_and_recommended = 0

strong_positive = 0

both_skin_fields_available = 0

earliest_date = None
latest_date = None


# ================================================================
# 3. PROCESS REVIEW FILES IN CHUNKS
# ================================================================

for file in review_files:

    print("\nProcessing:", file.name)

    for chunk in pd.read_csv(
        file,
        usecols=review_columns,
        chunksize=100_000,
        low_memory=False,
    ):

        total_reviews += len(chunk)


        # ----------------------------------------------------------
        # Normalize IDs
        # ----------------------------------------------------------

        chunk["author_id"] = (
            chunk["author_id"]
            .astype("string")
            .str.strip()
        )

        chunk["product_id"] = (
            chunk["product_id"]
            .astype("string")
            .str.strip()
        )


        # ----------------------------------------------------------
        # Unique users
        # ----------------------------------------------------------

        valid_users = (
            chunk["author_id"]
            .dropna()
            .astype(str)
        )

        unique_users.update(valid_users)


        # ----------------------------------------------------------
        # Unique products
        # ----------------------------------------------------------

        valid_products = (
            chunk["product_id"]
            .dropna()
            .astype(str)
        )

        unique_products.update(valid_products)


        # ----------------------------------------------------------
        # Unique USER-PRODUCT pairs
        # ----------------------------------------------------------

        valid_pairs = chunk[
            chunk["author_id"].notna()
            & chunk["product_id"].notna()
        ][
            ["author_id", "product_id"]
        ]

        unique_user_product_pairs.update(
            zip(
                valid_pairs["author_id"].astype(str),
                valid_pairs["product_id"].astype(str),
            )
        )


        # ----------------------------------------------------------
        # User activity
        # ----------------------------------------------------------

        user_interaction_counter.update(
            valid_users
        )


        # ----------------------------------------------------------
        # Product activity
        # ----------------------------------------------------------

        product_interaction_counter.update(
            valid_products
        )


        # ----------------------------------------------------------
        # Rating distribution
        # ----------------------------------------------------------

        rating_counter.update(
            chunk["rating"]
            .dropna()
            .astype(int)
            .tolist()
        )


        # ----------------------------------------------------------
        # Recommendation distribution
        # ----------------------------------------------------------

        recommended_counter.update(
            chunk["is_recommended"]
            .fillna("missing")
            .astype(str)
        )


        # ----------------------------------------------------------
        # Skin type distribution
        # ----------------------------------------------------------

        skin_type_counter.update(
            chunk["skin_type"]
            .fillna("missing")
            .astype(str)
        )


        # ----------------------------------------------------------
        # Skin tone distribution
        # ----------------------------------------------------------

        skin_tone_counter.update(
            chunk["skin_tone"]
            .fillna("missing")
            .astype(str)
        )


        # ----------------------------------------------------------
        # Missing values
        # ----------------------------------------------------------

        for column in review_columns:

            missing_counter[column] += (
                chunk[column]
                .isna()
                .sum()
            )


        # ----------------------------------------------------------
        # Positive interaction definitions
        # ----------------------------------------------------------

        rating_positive = (
            chunk["rating"] >= 4
        )

        recommended_positive = (
            chunk["is_recommended"] == 1
        )


        # Candidate definition 1
        positive_rating_only += (
            rating_positive.sum()
        )


        # Candidate definition 2
        positive_rating_and_recommended += (
            rating_positive
            & recommended_positive
        ).sum()


        # Very strong preference signal
        strong_positive += (
            (chunk["rating"] == 5)
            & recommended_positive
        ).sum()


        # ----------------------------------------------------------
        # Skin-profile availability
        # ----------------------------------------------------------

        both_skin_fields_available += (
            chunk["skin_type"].notna()
            & chunk["skin_tone"].notna()
        ).sum()


        # ----------------------------------------------------------
        # Time range
        # ----------------------------------------------------------

        dates = pd.to_datetime(
            chunk["submission_time"],
            errors="coerce",
        ).dropna()

        if not dates.empty:

            chunk_min = dates.min()
            chunk_max = dates.max()

            if (
                earliest_date is None
                or chunk_min < earliest_date
            ):
                earliest_date = chunk_min

            if (
                latest_date is None
                or chunk_max > latest_date
            ):
                latest_date = chunk_max


# ================================================================
# 4. GLOBAL REVIEW SUMMARY
# ================================================================

print_section("GLOBAL REVIEW SUMMARY")

print("Total review rows:", total_reviews)

print(
    "Unique user-product interactions:",
    len(unique_user_product_pairs),
)

print(
    "Duplicate user-product review rows:",
    total_reviews
    - len(unique_user_product_pairs),
)

print(
    "Unique users:",
    len(unique_users),
)

print(
    "Unique reviewed products:",
    len(unique_products),
)

print(
    "Earliest review:",
    earliest_date,
)

print(
    "Latest review:",
    latest_date,
)


# ================================================================
# 5. PRODUCT CATALOG OVERLAP
# ================================================================

print_section("CATALOG OVERLAP")


skincare_product_ids = set(
    skincare_products["product_id"]
    .dropna()
    .astype(str)
)


review_product_ids = set(
    unique_products
)


reviewed_skincare_products = (
    review_product_ids
    & skincare_product_ids
)


non_skincare_review_products = (
    review_product_ids
    - skincare_product_ids
)


print(
    "Skincare products in product_info:",
    len(skincare_product_ids),
)

print(
    "Products appearing in reviews:",
    len(review_product_ids),
)

print(
    "Reviewed products that are skincare:",
    len(reviewed_skincare_products),
)

print(
    "Reviewed products outside skincare:",
    len(non_skincare_review_products),
)


# ================================================================
# 6. MATRIX SPARSITY
# ================================================================

print_section("USER-PRODUCT MATRIX")


possible_interactions = (
    len(unique_users)
    * len(unique_products)
)


actual_interactions = (
    len(unique_user_product_pairs)
)


if possible_interactions > 0:

    density = (
        actual_interactions
        / possible_interactions
    )

    sparsity = 1 - density

    print(
        f"Matrix density: "
        f"{density:.8%}"
    )

    print(
        f"Matrix sparsity: "
        f"{sparsity:.8%}"
    )


# ================================================================
# 7. USER ACTIVITY
# ================================================================

print_section("USER ACTIVITY")


user_counts = np.array(
    list(
        user_interaction_counter.values()
    )
)


print(
    "Average reviews per user:",
    user_counts.mean(),
)

print(
    "Median reviews per user:",
    np.median(user_counts),
)

print(
    "Maximum reviews by one user:",
    user_counts.max(),
)


for threshold in [
    2,
    3,
    5,
    10,
    20,
    50,
]:

    print(
        f"Users with >= {threshold} reviews:",
        np.sum(
            user_counts >= threshold
        ),
    )


# ================================================================
# 8. PRODUCT ACTIVITY
# ================================================================

print_section("PRODUCT ACTIVITY")


product_counts = np.array(
    list(
        product_interaction_counter.values()
    )
)


print(
    "Average reviews per product:",
    product_counts.mean(),
)

print(
    "Median reviews per product:",
    np.median(product_counts),
)

print(
    "Maximum reviews for one product:",
    product_counts.max(),
)


# ================================================================
# 9. RATING DISTRIBUTION
# ================================================================

print_section("RATING DISTRIBUTION")


for rating in sorted(
    rating_counter.keys()
):

    count = rating_counter[rating]

    percentage = (
        count / total_reviews
    ) * 100

    print(
        f"{rating} stars: "
        f"{count:,} "
        f"({percentage:.2f}%)"
    )


# ================================================================
# 10. IS_RECOMMENDED
# ================================================================

print_section(
    "IS_RECOMMENDED DISTRIBUTION"
)


for key, value in (
    recommended_counter
    .most_common()
):

    percentage = (
        value / total_reviews
    ) * 100

    print(
        key,
        ":",
        f"{value:,}",
        f"({percentage:.2f}%)",
    )


# ================================================================
# 11. SKIN TYPE
# ================================================================

print_section(
    "SKIN TYPE DISTRIBUTION"
)


for key, value in (
    skin_type_counter
    .most_common()
):

    percentage = (
        value / total_reviews
    ) * 100

    print(
        key,
        ":",
        f"{value:,}",
        f"({percentage:.2f}%)",
    )


# ================================================================
# 12. SKIN TONE
# ================================================================

print_section(
    "SKIN TONE DISTRIBUTION"
)


for key, value in (
    skin_tone_counter
    .most_common()
):

    percentage = (
        value / total_reviews
    ) * 100

    print(
        key,
        ":",
        f"{value:,}",
        f"({percentage:.2f}%)",
    )


# ================================================================
# 13. PROFILE COVERAGE
# ================================================================

print_section(
    "SKIN PROFILE COVERAGE"
)


print(
    "Reviews with BOTH skin_type and skin_tone:",
    f"{both_skin_fields_available:,}",
)


print(
    "Percentage with BOTH:",
    f"{both_skin_fields_available / total_reviews * 100:.2f}%",
)


# ================================================================
# 14. REVIEW MISSING VALUES
# ================================================================

print_section(
    "MISSING REVIEW VALUES"
)


for column in review_columns:

    missing = missing_counter[column]

    percentage = (
        missing / total_reviews
    ) * 100

    print(
        f"{column:20}"
        f"{missing:10} "
        f"({percentage:.2f}%)"
    )


# ================================================================
# 15. POSITIVE INTERACTION ANALYSIS
# ================================================================

print_section(
    "POSITIVE INTERACTION ANALYSIS"
)


print(
    "Rating >= 4:",
    f"{positive_rating_only:,}",
)

print(
    "Percentage:",
    f"{positive_rating_only / total_reviews * 100:.2f}%",
)


print()


print(
    "Rating >= 4 AND is_recommended == 1:",
    f"{positive_rating_and_recommended:,}",
)

print(
    "Percentage:",
    f"{positive_rating_and_recommended / total_reviews * 100:.2f}%",
)


print()


print(
    "Rating == 5 AND is_recommended == 1:",
    f"{strong_positive:,}",
)

print(
    "Percentage:",
    f"{strong_positive / total_reviews * 100:.2f}%",
)