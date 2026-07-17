"""Mine frequent itemsets and association rules using Apriori."""

from pathlib import Path
import json
import random

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
TEST_SIZE = 0.20
MIN_SUPPORT = 0.08
MIN_CONFIDENCE = 0.35

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
METRICS_DIR = BASE_DIR / "outputs" / "metrics"
MODELS_DIR = BASE_DIR / "models"

TRAIN_PATH = DATA_DIR / "train_transactions.csv"
TEST_PATH = DATA_DIR / "test_transactions.csv"

FREQUENT_ITEMSETS_PATH = (
    MODELS_DIR / "frequent_itemsets.csv"
)
RULES_PATH = MODELS_DIR / "association_rules.csv"
TRAINING_SUMMARY_PATH = (
    METRICS_DIR / "training_summary.json"
)


def generate_transactions(
    transaction_count: int = 500,
) -> list[list[str]]:
    """Generate reproducible grocery transactions with patterns."""

    random_generator = random.Random(RANDOM_STATE)

    products = [
        "bread",
        "milk",
        "butter",
        "eggs",
        "cheese",
        "coffee",
        "tea",
        "sugar",
        "rice",
        "chicken",
        "pasta",
        "tomato_sauce",
        "apples",
        "bananas",
        "yogurt",
    ]

    transactions: list[list[str]] = []

    for _ in range(transaction_count):
        basket: set[str] = set()

        # General independent purchases
        for product in products:
            if random_generator.random() < 0.10:
                basket.add(product)

        # Embedded market-basket patterns
        if random_generator.random() < 0.40:
            basket.update(["bread", "milk"])

        if random_generator.random() < 0.30:
            basket.update(["bread", "butter"])

        if random_generator.random() < 0.25:
            basket.update(["coffee", "sugar"])

        if random_generator.random() < 0.23:
            basket.update(["pasta", "tomato_sauce"])

        if random_generator.random() < 0.20:
            basket.update(["rice", "chicken"])

        if random_generator.random() < 0.18:
            basket.update(["bananas", "yogurt"])

        # Ensure that no transaction is empty.
        if not basket:
            basket.add(
                random_generator.choice(products)
            )

        transactions.append(sorted(basket))

    return transactions


def split_transactions(
    transactions: list[list[str]],
) -> tuple[list[list[str]], list[list[str]]]:
    """Create training and held-out test transaction sets."""

    train_transactions, test_transactions = (
        train_test_split(
            transactions,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            shuffle=True,
        )
    )

    return train_transactions, test_transactions


def save_transactions(
    transactions: list[list[str]],
    output_path: Path,
) -> None:
    """Save transactions in basket format."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    transaction_table = pd.DataFrame(
        {
            "transaction_id": range(
                1,
                len(transactions) + 1,
            ),
            "items": [
                "|".join(transaction)
                for transaction in transactions
            ],
        }
    )

    transaction_table.to_csv(
        output_path,
        index=False,
    )


def encode_transactions(
    transactions: list[list[str]],
) -> pd.DataFrame:
    """Convert transaction baskets into Boolean columns."""

    encoder = TransactionEncoder()

    encoded_array = encoder.fit(
        transactions
    ).transform(transactions)

    return pd.DataFrame(
        encoded_array,
        columns=encoder.columns_,
    )


def itemset_to_text(itemset: frozenset) -> str:
    """Convert a frozenset into stable pipe-separated text."""

    return "|".join(sorted(itemset))


def mine_patterns(
    encoded_train: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mine frequent itemsets and association rules."""

    frequent_itemsets = apriori(
        encoded_train,
        min_support=MIN_SUPPORT,
        use_colnames=True,
        max_len=3,
    )

    frequent_itemsets["itemset_length"] = (
        frequent_itemsets["itemsets"].apply(len)
    )

    frequent_itemsets = frequent_itemsets.sort_values(
        by=["support", "itemset_length"],
        ascending=[False, True],
    )

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=MIN_CONFIDENCE,
    )

    rules = rules.sort_values(
        by=["lift", "confidence", "support"],
        ascending=False,
    )

    return frequent_itemsets, rules


def save_patterns(
    frequent_itemsets: pd.DataFrame,
    rules: pd.DataFrame,
) -> None:
    """Save mined itemsets and rules in readable form."""

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    itemset_output = frequent_itemsets.copy()
    itemset_output["itemsets"] = (
        itemset_output["itemsets"].apply(
            itemset_to_text
        )
    )

    itemset_output.to_csv(
        FREQUENT_ITEMSETS_PATH,
        index=False,
    )

    rule_output = rules.copy()

    rule_output["antecedents"] = (
        rule_output["antecedents"].apply(
            itemset_to_text
        )
    )

    rule_output["consequents"] = (
        rule_output["consequents"].apply(
            itemset_to_text
        )
    )

    rule_output.to_csv(
        RULES_PATH,
        index=False,
    )


def save_summary(
    train_transactions: list[list[str]],
    test_transactions: list[list[str]],
    encoded_train: pd.DataFrame,
    frequent_itemsets: pd.DataFrame,
    rules: pd.DataFrame,
) -> None:
    """Save configuration and training results."""

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {
        "algorithm": "Apriori",
        "application": "Market basket analysis",
        "total_transactions": (
            len(train_transactions)
            + len(test_transactions)
        ),
        "training_transactions": len(
            train_transactions
        ),
        "held_out_test_transactions": len(
            test_transactions
        ),
        "unique_items": int(
            encoded_train.shape[1]
        ),
        "minimum_support": MIN_SUPPORT,
        "minimum_confidence": MIN_CONFIDENCE,
        "maximum_itemset_length": 3,
        "frequent_itemset_count": len(
            frequent_itemsets
        ),
        "association_rule_count": len(rules),
        "rules_mined_from_training_only": True,
        "test_transactions_used_during_mining": False,
        "random_state": RANDOM_STATE,
    }

    with TRAINING_SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
        )


def main() -> None:
    """Run the complete Apriori mining workflow."""

    transactions = generate_transactions()

    train_transactions, test_transactions = (
        split_transactions(transactions)
    )

    save_transactions(
        train_transactions,
        TRAIN_PATH,
    )

    save_transactions(
        test_transactions,
        TEST_PATH,
    )

    encoded_train = encode_transactions(
        train_transactions
    )

    frequent_itemsets, rules = mine_patterns(
        encoded_train
    )

    save_patterns(
        frequent_itemsets,
        rules,
    )

    save_summary(
        train_transactions=train_transactions,
        test_transactions=test_transactions,
        encoded_train=encoded_train,
        frequent_itemsets=frequent_itemsets,
        rules=rules,
    )

    print("\nApriori mining completed.")
    print(
        f"Training transactions: "
        f"{len(train_transactions)}"
    )
    print(
        f"Held-out transactions: "
        f"{len(test_transactions)}"
    )
    print(
        f"Unique products: "
        f"{encoded_train.shape[1]}"
    )
    print(
        f"Frequent itemsets: "
        f"{len(frequent_itemsets)}"
    )
    print(
        f"Association rules: {len(rules)}"
    )
    print(f"Rules saved to: {RULES_PATH}")


if __name__ == "__main__":
    main()