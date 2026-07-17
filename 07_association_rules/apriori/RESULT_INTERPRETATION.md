
The support for a rule’s combined itemset, its antecedent support, and consequent support are distinct quantities; confidence and lift then describe conditional strength and departure from independence. :contentReference[oaicite:1]{index=1}

# Step 8 — Add interpretation file

Place this in `RESULT_INTERPRETATION.md`:

```markdown
# Apriori — Result Interpretation

## Purpose

This document explains the Apriori frequent-itemset and association-rule outputs.

## Frequent Itemsets

A frequent itemset is an item combination whose support is at least the configured minimum-support threshold.

Frequent does not necessarily mean useful or causal.

## Support

Support measures how frequently the combined itemset occurs across all transactions.

Higher support indicates a more common pattern.

Very low-support rules may be unstable or operationally unimportant.

## Confidence

Confidence answers:

> When the antecedent occurs, how often does the consequent also occur?

High confidence may still be misleading when the consequent is already extremely common.

## Lift

Lift compares observed rule confidence with the consequent’s general frequency.

- `lift > 1`: positive association;
- `lift = 1`: approximate independence;
- `lift < 1`: negative association.

Lift does not prove that one product causes the purchase of another.

## Held-Out Evaluation

Rules were mined from training transactions only.

Held-out support and confidence indicate whether those patterns remain reasonably stable on unseen baskets.

Large train-test differences may indicate:

- unstable rules;
- insufficient transactions;
- overly low support thresholds;
- random variation;
- changing customer behaviour.

## Recommendation Example

The prediction script selects one held-out basket.

Rules whose antecedents are contained in that basket generate candidate product recommendations.

Items already present in the basket are removed from the recommendation list.

## Limitations

Association rules:

- identify co-occurrence rather than causality;
- can produce many redundant rules;
- are sensitive to support and confidence thresholds;
- may favour popular products;
- can change as transaction patterns change.

## Final Conclusion

Apriori demonstrates:

- transactional-data encoding;
- frequent-itemset mining;
- support, confidence, and lift;
- training-only rule discovery;
- held-out rule validation;
- basket-based recommendations.