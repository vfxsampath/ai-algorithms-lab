# Content-Based Filtering — Result Interpretation

## 1. Purpose

This document explains the item vectors, user profiles, rankings, evaluation metrics, and limitations of the content-based recommendation system.

## 2. Item Representation

Each item is represented through a combined text document containing:

- title;
- category;
- difficulty;
- description;
- skills.

TF-IDF transforms this document into a numerical vector.

Items sharing important words and phrases receive more similar vectors.

## 3. TF-IDF Feature

Each TF-IDF feature corresponds to a learned term or phrase.

Examples include:

- `python`;
- `machine learning`;
- `agentic ai`;
- `process mining`;
- `digital transformation`.

A high TF-IDF value means the term is important to that specific item relative to the catalogue.

## 4. User Profile

The user profile combines the vectors of previously interacted items.

Interaction weights determine how strongly each item contributes.

Completed items contribute more than liked items, and liked items contribute more than viewed items.

## 5. Profile Normalization

The final user profile is normalized.

This allows cosine similarity to focus primarily on content direction instead of vector magnitude.

## 6. Similarity Score

The similarity score measures content alignment between the user profile and an unseen item.

A larger score indicates stronger shared content.

The score is not:

- a probability of purchase;
- a predicted rating;
- a guarantee of satisfaction;
- evidence of causality.

## 7. Recommendation Rank

Items are ordered from highest to lowest similarity.

Rank `1` is the most content-similar unseen item.

A small difference between two scores may not represent a meaningful preference difference.

## 8. Held-Out Evaluation

The latest positive user interaction is removed before building the user profile.

The recommender then attempts to recover that item in its ranked list.

This simulates whether earlier interests could identify a later relevant item.

## 9. Precision@K

Precision@K measures the relevant proportion of the first K recommendations.

With one held-out relevant item:

```text
Precision@5 =
1 / 5
```

when the item appears in the Top 5.

Otherwise, Precision@5 is zero.

## 10. Recall@K

Recall@K measures whether the held-out relevant item was recovered.

With one held-out item:

- `1` means it appeared;
- `0` means it did not.

## 11. Hit Rate@K

Hit Rate@K averages the user-level hits.

Example:

```text
8 users receive a relevant Top-5 item
out of 12 evaluated users

Hit Rate@5 = 8 / 12
```

## 12. Mean Reciprocal Rank

Reciprocal rank is:

```text
1 / rank of held-out item
```

Examples:

| Rank | Reciprocal Rank |
|---:|---:|
| 1 | 1.00 |
| 2 | 0.50 |
| 5 | 0.20 |
| Not ranked | 0.00 |

Mean Reciprocal Rank averages this value across users.

## 13. Catalogue Coverage

Catalogue coverage measures how broadly the recommender uses available items.

Low coverage may indicate:

- popularity concentration;
- very similar profiles;
- metadata imbalance;
- insufficient content variation.

High coverage does not automatically mean recommendations are relevant.

## 14. Intra-List Diversity

Intra-list diversity compares recommended items with one another.

Higher diversity means the recommendation list contains more varied content.

Very high diversity can reduce relevance.

A useful system balances:

- relevance;
- diversity;
- novelty;
- coverage.

## 15. Seen Items

Training items are excluded from the recommendation list.

This is suitable when the objective is discovery of new courses.

Some applications may intentionally recommend previously consumed items, such as music replay or refresher learning.

## 16. Cold-Start Items

Content-based filtering can recommend new items as soon as their metadata are available.

No interaction history is required for the item.

This is an advantage over pure collaborative filtering.

## 17. Cold-Start Users

A user with no interactions has no behavioural profile.

Possible solutions include:

- onboarding questionnaires;
- selected interests;
- popular-item recommendations;
- contextual information;
- demographic information where appropriate and lawful;
- hybrid recommendation.

## 18. Over-Specialization

Content-based systems can repeatedly recommend items highly similar to prior choices.

This can create a narrow recommendation bubble.

Possible improvements include diversity re-ranking and exploration rules.

## 19. TF-IDF Limitation

TF-IDF mainly represents word occurrence.

It may not recognize that semantically related phrases use different vocabulary.

For example:

```text
autonomous agent
```

and:

```text
goal-driven AI assistant
```

may describe similar concepts but share few terms.

Semantic embeddings can improve this.

## 20. Interaction Weights

The selected weights are design assumptions:

```text
viewed = 1
liked = 2
completed = 3
```

Real weights should be selected using user behaviour and business outcomes.

Completion is not always stronger evidence than liking.

## 21. Synthetic Data Limitation

The catalogue and interactions are generated for education.

The metrics demonstrate the evaluation workflow.

They do not establish performance for a real course platform.

## 22. One-User Example

`predict.py` displays:

- training interactions;
- interaction weights;
- held-out item;
- Top-5 recommendations;
- similarity scores;
- whether the held-out item appears.

This demonstrates the full recommendation process for one user.

## 23. Content-Based vs Collaborative Filtering

| Aspect | Content-Based | Collaborative |
|---|---|---|
| Main information | Item attributes | User-item behaviour |
| New-item handling | Strong | Weak |
| New-user handling | Weak | Weak |
| Explainability | Usually higher | Often lower |
| Cross-domain discovery | Limited | Potentially stronger |
| Metadata required | Yes | No |
| Community preferences | Not used | Central |

## 24. Recommended Improvements

1. Replace TF-IDF with sentence embeddings.
2. Compare TF-IDF and semantic retrieval.
3. Add recency weighting.
4. Add negative-feedback handling.
5. Add category-diversity re-ranking.
6. Add novelty measurement.
7. Add user onboarding preferences.
8. Add real interaction data.
9. Add API-based recommendations.
10. Add hybrid filtering.
11. Add online A/B testing.
12. Add recommendation explanations.

## 25. Final Conclusion

Content-based filtering recommends unseen items by comparing item features with a user's historical preference profile.

This project demonstrates:

- TF-IDF item representation;
- cosine similarity;
- weighted user profiles;
- chronological interaction holdout;
- Top-K ranking;
- Precision@K;
- Recall@K;
- Hit Rate@K;
- Mean Reciprocal Rank;
- catalogue coverage;
- recommendation diversity;
- one-user inference.

The next recommendation algorithm should be collaborative filtering, which learns from relationships across users rather than relying primarily on item descriptions.