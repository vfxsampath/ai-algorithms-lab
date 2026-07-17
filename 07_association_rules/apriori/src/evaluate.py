"""Evaluate mined association rules on held-out transactions."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

TEST_PATH = BASE_DIR / "data" / "test_transactions.csv"
RULES_PATH = (
    BASE_DIR
    / "models"
    / "association_rules.csv"
)

METRICS_DIR = BASE_DIR / "outputs" / "metrics"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"
PREDICTIONS_DIR = (
    BASE_DIR / "outputs" / "predictions"
)


def load_transactions(
    path: Path,
) -> list[set[str]]:
    """Load transaction baskets from CSV."""

    if not path.exists():
        raise FileNotFoundError(
            f"Transaction file not found: {path}"
        )

    table = pd.read_csv(path)

    return [
        set(str(items).split("|"))
        for items in table["items"]
    ]


def text_to_itemset(value: str) -> set[str]:
    """Convert stored text back to an item set."""

    return set(str(value).split("|"))


def load_rules() -> pd.DataFrame:
    """Load rules discovered from training data."""

    if not RULES_PATH.exists():
        raise FileNotFoundError(
            "Association rules not found. "
            "Run train.py first."
        )

    rules = pd.read_csv(RULES_PATH)

    if rules.empty:
        raise ValueError(
            "No rules were generated. "
            "Review support and confidence settings."
        )

    return rules


def calculate_held_out_metrics(
    rules: pd.DataFrame,
    test_transactions: list[set[str]],
) -> pd.DataFrame:
    """Calculate support and confidence on held-out data."""

    row_count = len(test_transactions)
    evaluated_rows = []

    for _, rule in rules.iterrows():
        antecedent = text_to_itemset(
            rule["antecedents"]
        )

        consequent = text_to_itemset(
            rule["consequents"]
        )

        combined = antecedent | consequent

        antecedent_count = sum(
            antecedent.issubset(transaction)
            for transaction in test_transactions
        )

        consequent_count = sum(
            consequent.issubset(transaction)
            for transaction in test_transactions
        )

        combined_count = sum(
            combined.issubset(transaction)
            for transaction in test_transactions
        )

        test_antecedent_support = (
            antecedent_count / row_count
        )

        test_consequent_support = (
            consequent_count / row_count
        )

        test_support = combined_count / row_count

        test_confidence = (
            combined_count / antecedent_count
            if antecedent_count > 0
            else 0.0
        )

        test_lift = (
            test_confidence
            / test_consequent_support
            if test_consequent_support > 0
            else 0.0
        )

        evaluated_rows.append(
            {
                "antecedents": rule[
                    "antecedents"
                ],
                "consequents": rule[
                    "consequents"
                ],
                "train_support": rule["support"],
                "test_support": test_support,
                "support_difference": (
                    test_support
                    - rule["support"]
                ),
                "train_confidence": rule[
                    "confidence"
                ],
                "test_confidence": test_confidence,
                "confidence_difference": (
                    test_confidence
                    - rule["confidence"]
                ),
                "train_lift": rule["lift"],
                "test_lift": test_lift,
                "test_antecedent_count": (
                    antecedent_count
                ),
                "test_combined_count": (
                    combined_count
                ),
            }
        )

    return pd.DataFrame(evaluated_rows)


def save_metrics(
    evaluated_rules: pd.DataFrame,
) -> None:
    """Save overall held-out stability metrics."""

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    valid_rules = evaluated_rules[
        evaluated_rules[
            "test_antecedent_count"
        ] > 0
    ]

    metrics = {
        "evaluated_rule_count": int(
            len(evaluated_rules)
        ),
        "rules_with_test_antecedent": int(
            len(valid_rules)
        ),
        "mean_absolute_support_difference": float(
            evaluated_rules[
                "support_difference"
            ].abs().mean()
        ),
        "mean_absolute_confidence_difference": float(
            valid_rules[
                "confidence_difference"
            ].abs().mean()
        ),
        "mean_test_support": float(
            evaluated_rules[
                "test_support"
            ].mean()
        ),
        "mean_test_confidence": float(
            valid_rules[
                "test_confidence"
            ].mean()
        ),
        "mean_test_lift": float(
            valid_rules[
                "test_lift"
            ].mean()
        ),
        "test_rules_with_lift_above_one": int(
            (valid_rules["test_lift"] > 1)
            .sum()
        ),
    }

    with (
        METRICS_DIR / "metrics.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)


def save_rule_evaluation(
    evaluated_rules: pd.DataFrame,
) -> None:
    """Save all train and test rule statistics."""

    PREDICTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    evaluated_rules.sort_values(
        by=[
            "test_lift",
            "test_confidence",
            "test_support",
        ],
        ascending=False,
    ).to_csv(
        PREDICTIONS_DIR
        / "held_out_rule_evaluation.csv",
        index=False,
    )


def save_train_test_support_plot(
    evaluated_rules: pd.DataFrame,
) -> None:
    """Compare training and test support."""

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(8, 7))

    plt.scatter(
        evaluated_rules["train_support"],
        evaluated_rules["test_support"],
        alpha=0.7,
    )

    maximum = max(
        evaluated_rules["train_support"].max(),
        evaluated_rules["test_support"].max(),
    )

    plt.plot(
        [0, maximum],
        [0, maximum],
        linestyle="--",
    )

    plt.xlabel("Training Support")
    plt.ylabel("Held-Out Test Support")
    plt.title(
        "Apriori Rule Support Stability"
    )
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "train_test_support.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def save_rule_scatter(
    evaluated_rules: pd.DataFrame,
) -> None:
    """Plot test support, confidence, and lift."""

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(9, 7))

    scatter = plt.scatter(
        evaluated_rules["test_support"],
        evaluated_rules["test_confidence"],
        s=(
            evaluated_rules["test_lift"]
            .clip(lower=0)
            * 60
        ),
        c=evaluated_rules["test_lift"],
        alpha=0.7,
    )

    plt.colorbar(
        scatter,
        label="Held-Out Test Lift",
    )

    plt.xlabel("Held-Out Test Support")
    plt.ylabel("Held-Out Test Confidence")
    plt.title(
        "Held-Out Association Rule Quality"
    )
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "rule_quality.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def main() -> None:
    """Run held-out association-rule evaluation."""

    test_transactions = load_transactions(
        TEST_PATH
    )

    rules = load_rules()

    evaluated_rules = (
        calculate_held_out_metrics(
            rules=rules,
            test_transactions=test_transactions,
        )
    )

    save_metrics(evaluated_rules)
    save_rule_evaluation(evaluated_rules)

    save_train_test_support_plot(
        evaluated_rules
    )

    save_rule_scatter(evaluated_rules)

    print("\nApriori Held-Out Evaluation")
    print(
        f"Rules evaluated: "
        f"{len(evaluated_rules)}"
    )

    top_rules = evaluated_rules.sort_values(
        by=[
            "test_lift",
            "test_confidence",
        ],
        ascending=False,
    ).head(10)

    print("\nTop held-out rules:")

    print(
        top_rules[
            [
                "antecedents",
                "consequents",
                "test_support",
                "test_confidence",
                "test_lift",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
