# K-Means Clustering — Result Interpretation

## 1. Purpose

This document explains how to interpret the K-Means clustering outputs.

K-Means is an unsupervised algorithm.

It does not predict known classes during training. Instead, it discovers groups according to similarities in the standardized feature space.

---

## 2. Meaning of Cluster Numbers

Cluster numbers such as:

- Cluster 0
- Cluster 1
- Cluster 2

are arbitrary identifiers.

Cluster 0 is not automatically better, larger, or more important than Cluster 1.

The business meaning of each cluster must be determined by examining its feature profile.

---

## 3. Cluster Sizes

The file:

```text
outputs/metrics/cluster_sizes.csv

shows the number of observations assigned to each cluster.

Very small clusters may represent:

unusual subgroups;
outliers;
unstable segmentation;
a genuinely specialized segment.

Large differences in cluster sizes should be investigated rather than automatically treated as errors.

4. Cluster Profiles

The file:

outputs/metrics/cluster_profiles.csv

contains the mean value of every original feature within each cluster.

These profiles are used to describe and name the clusters.

Possible descriptive labels might include:

high-intensity profile;
moderate-profile segment;
low-concentration segment.

Labels should be based on data and domain knowledge, not invented from cluster numbers.

5. Centroids

A centroid represents the average position of a cluster.

The file:

outputs/metrics/cluster_centroids.csv

contains centroid values transformed back into the original feature units.

A new observation is assigned to the cluster whose standardized centroid is closest.

Centroids are useful summaries, but not every cluster member will closely resemble the centroid.

6. Inertia

Inertia measures the total squared distance between observations and their assigned centroids.

Lower inertia indicates more compact clusters.

However, inertia always decreases when more clusters are added.

Therefore, inertia should not be used alone to select the cluster count.

7. Elbow Method

The elbow plot compares inertia across candidate values of k.

The preferred point is often where adding more clusters produces only a smaller additional reduction in inertia.

This bend is called the elbow.

The elbow may not always be visually clear.

Therefore, the elbow method should be combined with silhouette scores and domain interpretation.

8. Silhouette Score

The silhouette score evaluates:

cohesion within the assigned cluster;
separation from the nearest alternative cluster.

Its range is approximately:

Score	Interpretation
Near 1	Strong assignment
Around 0	Overlapping clusters
Below 0	Possibly assigned to the wrong cluster

Higher average scores generally indicate more clearly separated groups.

A higher silhouette score does not automatically make a clustering solution more useful for business purposes.

9. Sample Silhouette Scores

The file:

outputs/predictions/sample_silhouette_scores.csv

contains a silhouette score for every observation.

Observations with low or negative scores should be investigated.

They may lie:

near cluster boundaries;
between two plausible groups;
far from the main structure;
within poorly separated clusters.
10. Davies–Bouldin Score

The Davies–Bouldin score measures similarity among clusters.

Lower values are generally preferred.

A lower score suggests clusters are:

more compact internally;
more separated from one another.

It should be used as a comparative metric rather than interpreted as an absolute performance percentage.

11. Calinski–Harabasz Score

The Calinski–Harabasz score compares:

separation between clusters;
compactness within clusters.

Higher values are generally preferred when comparing clustering solutions on the same dataset.

It does not have a universal threshold for good clustering.

12. PCA Cluster Plot

The PCA visualization projects the standardized dataset into two dimensions.

The chart helps reveal:

visible cluster separation;
overlapping groups;
outlying observations;
centroid positions.

However, information is lost during dimensionality reduction.

Clusters that overlap in the two-dimensional PCA plot may still be separated in the complete feature space.

The K-Means model was trained using all standardized features, not only the two PCA components.

13. Selecting k

Candidate values from 2 through 10 are evaluated.

The final cluster count should consider:

Inertia and the elbow plot.
Mean silhouette score.
Davies–Bouldin score.
Calinski–Harabasz score.
Cluster-size balance.
Profile distinctiveness.
Business usefulness.
Stability across random samples.

The mathematically highest-scoring value is not always the most operationally meaningful choice.

14. Scaling Interpretation

K-Means was trained on standardized data.

This prevents features with large numerical ranges from dominating distance calculations.

Cluster profiles and centroids are transformed back to original units for human interpretation.

Scaling changes the geometry used during clustering but does not remove the original data meaning.

15. New Cluster Assignments

The prediction script assigns a new observation to the nearest learned centroid.

The output includes:

assigned cluster;
distance to the assigned centroid;
distances to all centroids.

A small distance indicates that the observation is relatively close to the cluster center.

A large distance may indicate:

an unusual observation;
weak cluster membership;
data drift;
a possible outlier.

K-Means does not produce class probabilities by default.

16. Business Interpretation

A useful cluster should have:

a distinctive profile;
sufficient membership;
operational relevance;
measurable differences from other clusters;
stability over time.

Cluster results might support:

differentiated marketing;
product strategy;
inventory planning;
customer segmentation;
service personalization;
anomaly investigation.

Clusters are analytical suggestions, not automatically valid business categories.

17. Key Assumptions

K-Means works best when clusters are approximately:

compact;
roughly spherical in feature space;
similar in spread;
separated by Euclidean distance.

It may perform poorly when clusters are:

irregularly shaped;
heavily overlapping;
strongly different in density;
strongly affected by outliers.
18. Comparison With Classification
Aspect	K-Means	Classification
Learning type	Unsupervised	Supervised
Target labels	Not used	Required
Main output	Cluster assignments	Predicted classes
Evaluation	Internal cluster metrics	Accuracy, recall, F1, ROC-AUC
Meaning of output	Must be interpreted	Defined by target labels
Typical use	Discovery and segmentation	Prediction

K-Means discovers structure.

Classification learns to reproduce known outcomes.

19. Recommended Improvements

Future improvements should include:

Compare different values of k.
Evaluate stability across random seeds.
Perform outlier treatment.
Compare scaled and unscaled results.
Compare K-Means with hierarchical clustering.
Compare K-Means with DBSCAN.
Add MiniBatchKMeans.
Add cluster-stability analysis.
Add interactive cluster-profile charts.
Add a FastAPI cluster-assignment endpoint.
Add automated tests.
Apply the method to a real customer-segmentation dataset.
20. Final Conclusion

K-Means provides an efficient centroid-based approach to discovering groups in numerical data.

Its main strengths are:

simplicity;
speed;
scalability;
clear centroid representation;
practical cluster assignment.

Its main weaknesses are:

required selection of k;
sensitivity to scaling;
sensitivity to outliers;
preference for compact cluster shapes;
dependence on interpretation.

In this repository, K-Means demonstrates:

unsupervised learning;
distance-based segmentation;
standardized feature spaces;
centroid interpretation;
internal clustering metrics;
cluster-count evaluation;
PCA-based visualization;
new-observation cluster assignment.