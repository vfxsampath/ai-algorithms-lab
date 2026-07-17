# Principal Component Analysis — Result Interpretation

## 1. Purpose

This document explains how to interpret the outputs produced by the PCA project.

PCA is an unsupervised dimensionality-reduction method.

It does not predict a target class or continuous target value.

Its purpose is to transform the original correlated feature space into a smaller component space while retaining as much variance as possible.

---

## 2. Train-Test Separation

The dataset was divided before preprocessing.

The training data were used to fit:

- the feature scaler;
- the principal components.

The held-out test data were not used during fitting.

The test data were transformed only after the pipeline was fitted.

This provides a more realistic evaluation of how the learned component structure behaves on unseen records.

---

## 3. Original Features and Retained Components

The original feature count is the number of variables in the source dataset.

The retained component count is the number of principal components needed to preserve the selected variance threshold.

For example:

```text
Original features: 13
Retained components: 10

This means that the dataset was represented using 10 component variables instead of 13 original variables.

4. Dimensionality-Reduction Percentage

Dimensionality reduction is calculated as:

Reduction percentage =
(1 - retained components / original features) × 100

A larger reduction means fewer component variables are required.

However, greater reduction generally causes greater information loss.

The objective is not necessarily to minimize the number of components. It is to find an appropriate balance between compression and retained information.

5. Explained Variance Ratio

Each principal component explains a proportion of total standardized variance.

For example:

PC1 explained variance ratio = 0.36

This means PC1 represents approximately 36% of the variation present in the standardized training dataset.

Components are ordered from highest to lowest explained variance.

6. Cumulative Explained Variance

Cumulative explained variance adds the explained variance ratios of successive components.

Example:

PC1 = 36%
PC1 + PC2 = 56%
PC1 + PC2 + PC3 = 68%

The pipeline retains the minimum number of components needed to reach at least 95% cumulative explained variance.

A high retained-variance percentage does not guarantee that every business-relevant pattern has been preserved.

7. Principal-Component Scores

The transformed train and test files contain component scores.

Each row represents one observation.

Each component column shows the observation’s position along that principal-component direction.

The scores no longer use the original feature units.

They are coordinates in the learned PCA space.

8. Component Loadings

Component loadings describe the contribution of each standardized original feature to a component.

A loading can be:

strongly positive;
strongly negative;
close to zero.

Large absolute values indicate stronger influence on that component.

The sign indicates direction, but PCA component signs can be reversed without changing the mathematical solution.

Therefore, the magnitude and pattern of loadings are usually more important than the sign alone.

9. Interpreting PC1

PC1 captures the greatest possible variance in the standardized training data.

To interpret PC1:

inspect features with the largest absolute PC1 loadings;
compare their signs;
identify the shared pattern represented by the component;
use domain knowledge to describe the pattern cautiously.

A component should not be assigned a business label without examining its loadings.

10. Interpreting PC2

PC2 captures the greatest remaining variance while being orthogonal to PC1.

PC2 represents a separate direction of variation.

It is interpreted by examining the largest absolute PC2 loadings.

PC2 does not mean the second-most important business concept. It means the second-largest variance direction under PCA’s mathematical objective.

11. Loading Heatmap

The loading heatmap displays:

original features along one axis;
retained components along the other;
component weights in the cells.

It helps identify:

features strongly associated with a component;
groups of features moving together;
opposite feature relationships;
weakly represented features.

Loadings do not establish causality.

12. Two-Component Projection

The train-test PCA projection uses PC1 and PC2 for visualization.

It helps reveal:

broad data structure;
possible groups;
unusual observations;
overlap between training and test distributions;
possible test records outside the training region.

However, PC1 and PC2 represent only part of the total variance.

Observations overlapping in the two-dimensional plot may differ in later components.

13. Training Reconstruction Error

Training observations are transformed into PCA components and reconstructed into the original feature space.

Training reconstruction error measures the loss created by dimensionality reduction on the data used to fit PCA.

Lower error means the retained components recreate the training observations more closely.

14. Test Reconstruction Error

Held-out test observations are transformed and reconstructed using the scaler and PCA components learned from training data.

Test reconstruction error indicates how well the learned component space represents unseen observations.

A moderate increase from training to test reconstruction error is expected.

A very large increase may indicate:

distribution differences;
outliers in the test data;
unstable component structure;
too few retained components;
weak generalization.
15. Original-Unit Reconstruction Error

Original-unit reconstruction error measures differences after transforming reconstructed values back to the original feature scales.

Features with large units may contribute heavily to this overall measure.

Therefore, original-unit reconstruction error should be interpreted together with standardized reconstruction error.

16. Standardized Reconstruction Error

Standardized reconstruction error is measured after all features have been placed on comparable scales.

This gives every standardized feature a more equal contribution.

It is often more appropriate for comparing overall PCA reconstruction quality.

17. Feature-Level Reconstruction Error

The file:

outputs/metrics/test_reconstruction_error_by_feature.csv

shows the held-out reconstruction error for each original feature.

Features with higher errors are less accurately represented by the retained components.

Possible reasons include:

feature-specific variation not captured by retained components;
outliers;
weak correlation with dominant feature patterns;
nonlinear relationships.

A high error does not mean the feature is unimportant.

18. Reconstruction Error and Information Loss

Dimensionality reduction is lossy unless all components are retained.

Some reconstruction error is expected.

The amount of acceptable information loss depends on the application.

For visualization, higher information loss may be acceptable.

For high-risk prediction or scientific measurement, stricter retention may be necessary.

19. External Data Transformation

The prediction script can transform external CSV records.

The external data must contain:

the exact expected feature names;
numerical values;
no missing values;
the same feature definitions and units used during training.

The saved training scaler is applied automatically.

The external data must not be fitted separately with a new scaler.

20. Held-Out Demo Sample

When no external CSV is supplied, predict.py loads one row from the saved held-out test dataset.

This row was not used to fit the scaler or PCA.

It is therefore a valid unseen demonstration sample for the fitted pipeline.

It is still drawn from the same original dataset and should not be described as independent real-world external validation.

21. PCA and Feature Scaling

The original features use different units and ranges.

PCA would otherwise give greater influence to high-variance, large-scale variables.

Standardization ensures that PCA is fitted using comparable feature scales.

The fitted training mean and standard deviation are reused for test and external data.

22. PCA and Correlation

PCA can summarize groups of correlated numerical features.

When multiple features carry similar information, a principal component may represent their shared variation.

This can reduce redundancy.

However, PCA does not identify causal relationships among features.

23. PCA and Multicollinearity

Principal components are orthogonal.

This means retained components are uncorrelated in the fitted training space.

Using principal components as inputs to later models can reduce multicollinearity.

The trade-off is reduced direct interpretability because the components are weighted combinations of original features.

24. PCA Before Clustering

PCA can be applied before K-Means or DBSCAN to:

reduce noise;
remove redundancy;
improve computational efficiency;
reduce high-dimensional distance problems;
support visualization.

However, clustering results should be compared with and without PCA.

PCA may remove lower-variance structure that is useful for segmentation.

25. PCA Before Classification or Regression

PCA can be included inside a supervised-learning pipeline.

Correct order:

Training data
→ Fit scaler
→ Fit PCA
→ Fit predictive model

For cross-validation, all these steps should remain inside one pipeline so preprocessing is fitted independently within each fold.

PCA should not be fitted on the full dataset before splitting or cross-validation.

26. Outliers

PCA can be influenced by extreme observations because they affect feature variance and covariance structure.

Outliers may rotate principal-component directions.

Possible improvements include:

robust outlier analysis;
RobustScaler;
robust PCA methods;
sensitivity analysis with and without unusual records.

Outliers should not be removed automatically without investigation.

27. Linearity Limitation

PCA captures linear combinations of features.

It may not represent nonlinear manifolds or curved structures effectively.

Alternatives include:

Kernel PCA;
t-SNE;
UMAP;
autoencoders.

These techniques have different objectives and interpretation limitations.

28. Comparison With Feature Selection

PCA creates new component variables.

Feature selection retains a subset of original variables.

Aspect	PCA	Feature Selection
Output	New components	Original features
Interpretability	Lower	Higher
Correlation reduction	Strong	Depends on method
Information combination	Yes	No
Original feature meanings	Mixed	Preserved
Reconstruction	Possible	Not equivalent

PCA is feature extraction, not ordinary feature selection.

29. Strengths

PCA provides:

dimensionality reduction;
linear data compression;
explained-variance reporting;
uncorrelated component variables;
visualization support;
approximate reconstruction;
reduced multicollinearity;
possible downstream efficiency improvements.
30. Limitations

Its main limitations are:

component interpretability;
linearity;
scaling sensitivity;
outlier sensitivity;
information loss;
dependence on the training distribution;
possible loss of low-variance but important information.
31. Recommended Improvements

Future improvements should include:

Compare 80%, 90%, 95%, and 99% variance thresholds.
Evaluate a fixed number of components.
Compare train and test reconstruction curves.
Add cross-validated downstream classification.
Compare clustering before and after PCA.
Add feature-biplot visualization.
Add outlier-sensitivity analysis.
Compare PCA with Kernel PCA.
Compare PCA with t-SNE and UMAP.
Add FastAPI transformation endpoint.
Add unit tests for feature validation.
Add automatic external batch processing.
32. Final Conclusion

PCA reduces the original numerical feature space by representing observations through principal components.

The model is fitted only on training data and evaluated by transforming held-out test observations.

Its main strengths are:

compression;
variance preservation;
correlation reduction;
visualization;
reusable feature transformation.

Its main weaknesses are:

information loss;
reduced interpretability;
linearity;
sensitivity to scaling and outliers.

In this repository, PCA demonstrates:

proper train-test preprocessing;
training-only scaler fitting;
training-only PCA fitting;
held-out transformation;
explained variance;
component loadings;
reconstruction error;
external CSV transformation.