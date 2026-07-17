"""Recommend products for one held-out transaction."""

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

TEST_PATH = BASE_DIR / "data" / "test_transactions.csv"
RULES_PATH = (
    BASE_DIR
    / "models"
    / "association_rules.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "outputs"
    / "predictions"
    / "example_recommendations.csv"
)


def load_test_basket() -> set[str]:
    """Load one basket from held-out test transactions."""

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            "Test transactions not found. "
            "Run train.py first."
        )

    test_data = pd.read_csv(TEST_PATH)

    if test_data.empty:
        raise ValueError(
            "Held-out test data are empty."
        )

    # Select a basket likely to activate a rule.
    for items in test_data["items"]:
        basket = set(str(items).split("|"))

        if len(basket) >= 2:
            return basket

    return set(
        str(test_data.iloc[0]["items"])
        .split("|")
    )


def load_rules() -> pd.DataFrame:
    """Load association rules mined from training data."""

    if not RULES_PATH.exists():
        raise FileNotFoundError(
            "Association rules not found. "
            "Run train.py first."
        )

    return pd.read_csv(RULES_PATH)


def parse_itemset(value: str) -> set[str]:
    """Convert pipe-separated text to a set."""

    return set(str(value).split("|"))


def recommend_items(
    basket: set[str],
    rules: pd.DataFrame,
) -> pd.DataFrame:
    """Apply matching association rules to a basket."""

    recommendations = []

    for _, rule in rules.iterrows():
        antecedent = parse_itemset(
            rule["antecedents"]
        )

        consequent = parse_itemset(
            rule["consequents"]
        )

        if not antecedent.issubset(basket):
            continue

        new_items = consequent - basket

        if not new_items:
            continue

        for item in new_items:
            recommendations.append(
                {
                    "recommended_item": item,
                    "triggering_items": "|".join(
                        sorted(antecedent)
                    ),
                    "confidence": rule[
                        "confidence"
                    ],
                    "lift": rule["lift"],
                    "support": rule["support"],
                }
            )

    if not recommendations:
        return pd.DataFrame(
            columns=[
                "recommended_item",
                "triggering_items",
                "confidence",
                "lift",
                "support",
            ]
        )

    recommendation_table = pd.DataFrame(
        recommendations
    )

    return (
        recommendation_table
        .sort_values(
            by=["lift", "confidence", "support"],
            ascending=False,
        )
        .drop_duplicates(
            subset=["recommended_item"],
            keep="first",
        )
        .reset_index(drop=True)
    )


def main() -> None:
    """Generate recommendations for one held-out basket."""

    basket = load_test_basket()
    rules = load_rules()

    recommendations = recommend_items(
        basket=basket,
        rules=rules,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    recommendations.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nApriori Recommendation Example")
    print(
        "Held-out basket: "
        f"{sorted(basket)}"
    )

    if recommendations.empty:
        print(
            "No matching recommendation rule "
            "was found for this basket."
        )
        return

    print("\nRecommended items:")

    print(
        recommendations.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
