"""Generate content-based recommendations for one evaluated user."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


EXAMPLE_USER_ID = 107
TOP_K = 5

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

OUTPUT_PATH = (
    BASE_DIR
    / "outputs"
    / "predictions"
    / "example_user_recommendations.csv"
)


def load_artifacts():
    """Load model and saved interaction data."""

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
        TRAIN_INTERACTIONS_PATH
    )

    test_interactions = pd.read_csv(
        TEST_INTERACTIONS_PATH
    )

    return (
        artifacts,
        items,
        train_interactions,
        test_interactions,
    )


def generate_recommendations(
    user_id: int,
    artifacts: dict,
    items: pd.DataFrame,
    train_interactions: pd.DataFrame,
) -> pd.DataFrame:
    """Rank unseen catalogue items for one user."""

    user_profiles = artifacts[
        "user_profiles"
    ]

    if user_id not in user_profiles:
        raise ValueError(
            f"User {user_id} does not have a saved profile."
        )

    item_ids = [
        int(value)
        for value in artifacts[
            "item_ids"
        ]
    ]

    user_profile = user_profiles[
        user_id
    ]

    item_matrix = artifacts[
        "item_matrix"
    ]

    scores = cosine_similarity(
        user_profile,
        item_matrix,
    ).ravel()

    seen_items = set(
        train_interactions.loc[
            train_interactions[
                "user_id"
            ] == user_id,
            "item_id",
        ].astype(int)
    )

    recommendation_table = pd.DataFrame(
        {
            "item_id": item_ids,
            "similarity_score": scores,
        }
    )

    recommendation_table = (
        recommendation_table[
            ~recommendation_table[
                "item_id"
            ].isin(seen_items)
        ]
        .sort_values(
            "similarity_score",
            ascending=False,
        )
        .head(TOP_K)
        .reset_index(drop=True)
    )

    recommendation_table[
        "rank"
    ] = range(
        1,
        len(recommendation_table) + 1,
    )

    return recommendation_table.merge(
        items,
        on="item_id",
        how="left",
    )


def main() -> None:
    """Run one held-out recommendation example."""

    (
        artifacts,
        items,
        train_interactions,
        test_interactions,
    ) = load_artifacts()

    user_id = EXAMPLE_USER_ID

    known_interactions = (
        train_interactions[
            train_interactions[
                "user_id"
            ] == user_id
        ]
        .merge(
            items[
                [
                    "item_id",
                    "title",
                    "category",
                ]
            ],
            on="item_id",
            how="left",
        )
        .sort_values(
            "interaction_weight",
            ascending=False,
        )
    )

    held_out = (
        test_interactions[
            test_interactions[
                "user_id"
            ] == user_id
        ]
        .merge(
            items[
                [
                    "item_id",
                    "title",
                    "category",
                ]
            ],
            on="item_id",
            how="left",
        )
    )

    recommendations = (
        generate_recommendations(
            user_id=user_id,
            artifacts=artifacts,
            items=items,
            train_interactions=(
                train_interactions
            ),
        )
    )

    held_out_item_id = int(
        held_out["item_id"].iloc[0]
    )

    recommendations[
        "is_held_out_item"
    ] = (
        recommendations["item_id"]
        == held_out_item_id
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    recommendations.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "\nContent-Based Recommendation Example"
    )

    print(
        f"\nUser ID: {user_id}"
    )

    print(
        "\nTraining interactions used "
        "to build the user profile:"
    )

    print(
        known_interactions[
            [
                "title",
                "category",
                "interaction_type",
                "interaction_weight",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nHeld-out relevant item:"
    )

    print(
        held_out[
            [
                "title",
                "category",
                "interaction_type",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        f"\nTop-{TOP_K} recommendations:"
    )

    print(
        recommendations[
            [
                "rank",
                "title",
                "category",
                "difficulty",
                "similarity_score",
                "is_held_out_item",
            ]
        ].to_string(
            index=False
        )
    )

    hit = bool(
        recommendations[
            "is_held_out_item"
        ].any()
    )

    print(
        f"\nHeld-out item found in "
        f"Top-{TOP_K}: {hit}"
    )

    print(
        "\nThe held-out interaction was "
        "not used to construct the user profile."
    )


if __name__ == "__main__":
    main()
