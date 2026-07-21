"""Evaluate content-based recommendations using held-out interactions."""

from __future__ import annotations

from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


TOP_K_VALUES = [3, 5, 10]

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "content_recommender.joblib"
)

ITEMS_PATH = (
    BASE_DIR
    / "data"
    / "items.csv"
)

TRAIN_INTERACTIONS_PATH = (
    BASE_DIR
    / "data"
    / "train_interactions.csv"
)

TEST_INTERACTIONS_PATH = (
    BASE_DIR
    / "data"
    / "test_interactions.csv"
)

FIGURES_DIR = (
    BASE_DIR
    / "outputs"
    / "figures"
)

METRICS_DIR = (
    BASE_DIR
    / "outputs"
    / "metrics"
)

PREDICTIONS_DIR = (
    BASE_DIR
    / "outputs"
    / "predictions"
)


def load_artifacts():
    """Load model artifacts and evaluation data."""

    required_paths = [
        MODEL_PATH,
        ITEMS_PATH,
        TRAIN_INTERACTIONS_PATH,
        TEST_INTERACTIONS_PATH,
    ]

    missing = [
        path
        for path in required_paths
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            f"Required files are missing: {missing}"
        )

    artifacts = joblib.load(
        MODEL_PATH
    )

    items = pd.read_csv(
        ITEMS_PATH
    )

    train_interactions = pd.read_csv(
        TRAIN_INTERACTIONS_PATH,
        parse_dates=[
            "interaction_date",
        ],
    )

    test_interactions = pd.read_csv(
        TEST_INTERACTIONS_PATH,
        parse_dates=[
            "interaction_date",
        ],
    )

    return (
        artifacts,
        items,
        train_interactions,
        test_interactions,
    )


def rank_items_for_user(
    user_id: int,
    user_profile: csr_matrix,
    item_matrix: csr_matrix,
    item_ids: list[int],
    seen_item_ids: set[int],
) -> pd.DataFrame:
    """Rank unseen items by cosine similarity."""

    similarity_scores = cosine_similarity(
        user_profile,
        item_matrix,
    ).ravel()

    rankings = pd.DataFrame(
        {
            "item_id": item_ids,
            "similarity_score": (
                similarity_scores
            ),
        }
    )

    rankings = rankings[
        ~rankings["item_id"].isin(
            seen_item_ids
        )
    ].copy()

    rankings["user_id"] = user_id

    rankings = rankings.sort_values(
        by="similarity_score",
        ascending=False,
    ).reset_index(drop=True)

    rankings["rank"] = np.arange(
        1,
        len(rankings) + 1,
    )

    return rankings


def calculate_user_metrics(
    user_id: int,
    ranked_items: pd.DataFrame,
    held_out_item_id: int,
) -> list[dict[str, float | int]]:
    """Calculate ranking metrics for one user."""

    matching_rows = ranked_items[
        ranked_items["item_id"]
        == held_out_item_id
    ]

    if matching_rows.empty:
        held_out_rank = None
        reciprocal_rank = 0.0
    else:
        held_out_rank = int(
            matching_rows["rank"].iloc[0]
        )

        reciprocal_rank = (
            1.0 / held_out_rank
        )

    rows = []

    for top_k in TOP_K_VALUES:
        top_items = ranked_items.head(
            top_k
        )["item_id"].tolist()

        hit = int(
            held_out_item_id in top_items
        )

        precision_at_k = (
            hit / top_k
        )

        recall_at_k = float(hit)

        rows.append(
            {
                "user_id": user_id,
                "top_k": top_k,
                "held_out_item_id": (
                    held_out_item_id
                ),
                "held_out_rank": (
                    held_out_rank
                    if held_out_rank is not None
                    else -1
                ),
                "hit": hit,
                "precision_at_k": (
                    precision_at_k
                ),
                "recall_at_k": (
                    recall_at_k
                ),
                "reciprocal_rank": (
                    reciprocal_rank
                ),
            }
        )

    return rows


def calculate_intra_list_diversity(
    recommended_item_ids: list[int],
    item_matrix: csr_matrix,
    item_position: dict[int, int],
) -> float:
    """Calculate average pairwise dissimilarity."""

    if len(recommended_item_ids) < 2:
        return 0.0

    positions = [
        item_position[item_id]
        for item_id in recommended_item_ids
    ]

    selected_matrix = item_matrix[
        positions
    ]

    similarity_matrix = cosine_similarity(
        selected_matrix
    )

    upper_indices = np.triu_indices(
        len(recommended_item_ids),
        k=1,
    )

    pairwise_similarities = (
        similarity_matrix[
            upper_indices
        ]
    )

    if len(pairwise_similarities) == 0:
        return 0.0

    return float(
        1.0
        - np.mean(
            pairwise_similarities
        )
    )


def evaluate_all_users(
    artifacts: dict,
    items: pd.DataFrame,
    train_interactions: pd.DataFrame,
    test_interactions: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Generate rankings and evaluate held-out items."""

    item_matrix = artifacts[
        "item_matrix"
    ]

    item_ids = [
        int(value)
        for value in artifacts[
            "item_ids"
        ]
    ]

    user_profiles = artifacts[
        "user_profiles"
    ]

    item_position = {
        item_id: position
        for position, item_id in enumerate(
            item_ids
        )
    }

    ranking_tables = []
    metric_rows = []

    for test_row in test_interactions.itertuples():
        user_id = int(test_row.user_id)

        held_out_item_id = int(
            test_row.item_id
        )

        user_profile = user_profiles[
            user_id
        ]

        seen_item_ids = set(
            train_interactions.loc[
                train_interactions[
                    "user_id"
                ] == user_id,
                "item_id",
            ].astype(int)
        )

        ranked_items = rank_items_for_user(
            user_id=user_id,
            user_profile=user_profile,
            item_matrix=item_matrix,
            item_ids=item_ids,
            seen_item_ids=seen_item_ids,
        )

        ranked_items[
            "held_out_item_id"
        ] = held_out_item_id

        ranked_items[
            "is_held_out_item"
        ] = (
            ranked_items["item_id"]
            == held_out_item_id
        )

        top_ten_ids = ranked_items.head(
            10
        )["item_id"].astype(int).tolist()

        diversity_at_10 = (
            calculate_intra_list_diversity(
                recommended_item_ids=(
                    top_ten_ids
                ),
                item_matrix=item_matrix,
                item_position=item_position,
            )
        )

        ranked_items[
            "diversity_at_10"
        ] = diversity_at_10

        ranking_tables.append(
            ranked_items
        )

        metric_rows.extend(
            calculate_user_metrics(
                user_id=user_id,
                ranked_items=ranked_items,
                held_out_item_id=(
                    held_out_item_id
                ),
            )
        )

    rankings = pd.concat(
        ranking_tables,
        ignore_index=True,
    )

    rankings = rankings.merge(
        items[
            [
                "item_id",
                "title",
                "category",
                "difficulty",
            ]
        ],
        on="item_id",
        how="left",
    )

    user_metrics = pd.DataFrame(
        metric_rows
    )

    return rankings, user_metrics


def calculate_catalogue_coverage(
    rankings: pd.DataFrame,
    catalogue_size: int,
    top_k: int,
) -> float:
    """Calculate the proportion of catalogue items recommended."""

    top_recommendations = (
        rankings
        .sort_values(
            [
                "user_id",
                "rank",
            ]
        )
        .groupby(
            "user_id",
            group_keys=False,
        )
        .head(top_k)
    )

    unique_recommended_items = (
        top_recommendations[
            "item_id"
        ].nunique()
    )

    return float(
        unique_recommended_items
        / catalogue_size
    )


def summarize_metrics(
    rankings: pd.DataFrame,
    user_metrics: pd.DataFrame,
    item_count: int,
) -> dict[str, float | int]:
    """Create aggregate recommendation metrics."""

    summary: dict[str, float | int] = {
        "evaluated_user_count": int(
            user_metrics[
                "user_id"
            ].nunique()
        ),
        "catalogue_item_count": int(
            item_count
        ),
    }

    for top_k in TOP_K_VALUES:
        metrics_at_k = user_metrics[
            user_metrics["top_k"]
            == top_k
        ]

        summary[
            f"precision_at_{top_k}"
        ] = float(
            metrics_at_k[
                "precision_at_k"
            ].mean()
        )

        summary[
            f"recall_at_{top_k}"
        ] = float(
            metrics_at_k[
                "recall_at_k"
            ].mean()
        )

        summary[
            f"hit_rate_at_{top_k}"
        ] = float(
            metrics_at_k[
                "hit"
            ].mean()
        )

        summary[
            f"catalogue_coverage_at_{top_k}"
        ] = calculate_catalogue_coverage(
            rankings=rankings,
            catalogue_size=item_count,
            top_k=top_k,
        )

    one_row_per_user = (
        user_metrics[
            user_metrics["top_k"]
            == TOP_K_VALUES[0]
        ]
    )

    summary[
        "mean_reciprocal_rank"
    ] = float(
        one_row_per_user[
            "reciprocal_rank"
        ].mean()
    )

    diversity_values = (
        rankings
        .groupby("user_id")[
            "diversity_at_10"
        ]
        .first()
    )

    summary[
        "mean_intra_list_diversity_at_10"
    ] = float(
        diversity_values.mean()
    )

    return summary


def save_outputs(
    rankings: pd.DataFrame,
    user_metrics: pd.DataFrame,
    metrics_summary: dict[str, float | int],
) -> None:
    """Save rankings and metric tables."""

    PREDICTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rankings.to_csv(
        PREDICTIONS_DIR
        / "all_user_rankings.csv",
        index=False,
    )

    user_metrics.to_csv(
        METRICS_DIR
        / "user_ranking_metrics.csv",
        index=False,
    )

    with (
        METRICS_DIR
        / "metrics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics_summary,
            file,
            indent=4,
        )


def save_metric_plot(
    metrics_summary: dict[str, float | int],
) -> None:
    """Plot hit rate and catalogue coverage."""

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    k_labels = [
        str(value)
        for value in TOP_K_VALUES
    ]

    hit_rates = [
        float(
            metrics_summary[
                f"hit_rate_at_{value}"
            ]
        )
        for value in TOP_K_VALUES
    ]

    coverage_values = [
        float(
            metrics_summary[
                f"catalogue_coverage_at_{value}"
            ]
        )
        for value in TOP_K_VALUES
    ]

    x_positions = np.arange(
        len(TOP_K_VALUES)
    )

    width = 0.35

    plt.figure(figsize=(9, 6))

    plt.bar(
        x_positions - width / 2,
        hit_rates,
        width,
        label="Hit Rate",
    )

    plt.bar(
        x_positions + width / 2,
        coverage_values,
        width,
        label="Catalogue Coverage",
    )

    plt.xticks(
        x_positions,
        k_labels,
    )

    plt.xlabel("K")
    plt.ylabel("Score")
    plt.title(
        "Recommendation Performance by K"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "performance_by_k.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def save_rank_distribution_plot(
    user_metrics: pd.DataFrame,
) -> None:
    """Plot the held-out item rank for each user."""

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ranks = (
        user_metrics[
            user_metrics["top_k"]
            == TOP_K_VALUES[0]
        ][
            [
                "user_id",
                "held_out_rank",
            ]
        ]
        .sort_values("user_id")
    )

    plt.figure(figsize=(10, 6))

    plt.bar(
        ranks[
            "user_id"
        ].astype(str),
        ranks[
            "held_out_rank"
        ],
    )

    plt.xlabel("User")
    plt.ylabel("Held-Out Item Rank")
    plt.title(
        "Held-Out Recommendation Rank by User"
    )
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "held_out_rank_by_user.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def save_recommendation_category_plot(
    rankings: pd.DataFrame,
) -> None:
    """Plot category distribution among Top-5 recommendations."""

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    top_five = (
        rankings
        .sort_values(
            [
                "user_id",
                "rank",
            ]
        )
        .groupby(
            "user_id",
            group_keys=False,
        )
        .head(5)
    )

    category_counts = (
        top_five[
            "category"
        ]
        .value_counts()
        .sort_values()
    )

    plt.figure(figsize=(10, 7))

    plt.barh(
        category_counts.index,
        category_counts.values,
    )

    plt.xlabel(
        "Number of Top-5 Recommendations"
    )

    plt.ylabel("Category")

    plt.title(
        "Recommendation Category Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "recommendation_categories.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def main() -> None:
    """Run complete held-out recommendation evaluation."""

    (
        artifacts,
        items,
        train_interactions,
        test_interactions,
    ) = load_artifacts()

    rankings, user_metrics = (
        evaluate_all_users(
            artifacts=artifacts,
            items=items,
            train_interactions=(
                train_interactions
            ),
            test_interactions=(
                test_interactions
            ),
        )
    )

    metrics_summary = summarize_metrics(
        rankings=rankings,
        user_metrics=user_metrics,
        item_count=len(items),
    )

    save_outputs(
        rankings=rankings,
        user_metrics=user_metrics,
        metrics_summary=metrics_summary,
    )

    save_metric_plot(
        metrics_summary
    )

    save_rank_distribution_plot(
        user_metrics
    )

    save_recommendation_category_plot(
        rankings
    )

    print(
        "\nContent-Based Recommendation "
        "Evaluation"
    )

    for metric_name, metric_value in (
        metrics_summary.items()
    ):
        if isinstance(
            metric_value,
            float,
        ):
            print(
                f"{metric_name}: "
                f"{metric_value:.4f}"
            )
        else:
            print(
                f"{metric_name}: "
                f"{metric_value}"
            )

    print(
        "\nEvaluation outputs saved."
    )


if __name__ == "__main__":
    main()
