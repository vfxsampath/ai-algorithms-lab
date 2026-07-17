# Apriori Association-Rule Mining

## Overview

Apriori discovers frequent item combinations and association rules from transactional data.

This implementation demonstrates market-basket analysis using separate training and held-out transaction sets.

## Business Problem

A retailer wants to discover products commonly purchased together and use those patterns for:

- cross-selling;
- product recommendations;
- bundle design;
- shelf-layout planning;
- promotion design;
- customer-basket analysis.

## Train-Test Design

Rules are mined only from training transactions.

Held-out transactions are used to determine whether training support, confidence, and lift remain stable on unseen baskets.

## Main Measures

### Support

The percentage of all transactions containing the complete item combination.

### Confidence

How often the consequent appears when the antecedent appears.

### Lift

How much more often the items occur together than expected under independence.

- Lift above `1`: positive association
- Lift near `1`: approximately independent
- Lift below `1`: negative association

## Run

```powershell
python 07_association_rules/apriori/src/train.py
python 07_association_rules/apriori/src/evaluate.py
python 07_association_rules/apriori/src/predict.py