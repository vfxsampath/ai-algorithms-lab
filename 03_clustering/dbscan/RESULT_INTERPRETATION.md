# DBSCAN Clustering — Result Interpretation

## 1. Purpose

This document explains how to interpret the outputs produced by the DBSCAN clustering project.

DBSCAN is an unsupervised density-based algorithm.

It discovers dense groups without using target labels and explicitly identifies observations that do not belong to a sufficiently dense region.

---

## 2. Cluster Labels

DBSCAN assigns ordinary clusters numerical labels such as:

- Cluster 0
- Cluster 1
- Cluster 2

These numbers are arbitrary identifiers.

They do not indicate rank, value, quality, or importance.

DBSCAN assigns:

```text
-1
to noise observations.

3. Core Samples

Core samples are observations located in sufficiently dense neighborhoods.

An observation becomes a core sample when its eps neighborhood contains at least min_samples observations, including itself.

Core samples form the internal structure of a DBSCAN cluster.

A high core-sample count suggests that many observations lie within dense regions.

4. Border Samples

Border samples are close enough to a core sample to belong to its cluster but do not independently satisfy the density requirement.

They lie near the outside of dense cluster regions.

Border observations may be less strongly embedded in a cluster than core observations.

5. Noise Samples

Noise observations are assigned cluster label:

-1

A noise observation does not fall within the reachable dense region of a cluster.

Noise may represent:

true outliers;
rare product profiles;
unusual customer behaviour;
measurement errors;
weakly connected observations;
observations rejected because parameters are too strict.

Noise should be investigated rather than automatically deleted.

6. Noise Percentage

Noise percentage is calculated as:

Noise percentage =
Noise observations /
All observations × 100

A very high noise percentage may indicate:

eps is too small;
min_samples is too high;
data contain weak density structure;
dimensionality is affecting distances;
observations genuinely contain many unusual profiles.

A very low noise percentage may indicate:

eps is too large;
clusters are being merged;
the model is not distinguishing isolated observations.

There is no universal ideal noise percentage.

7. Interpretation of eps

The model uses:

eps = 2.3

eps defines the radius of an observation’s neighborhood in standardized feature space.

A smaller eps normally produces:

smaller neighborhoods;
fewer core samples;
more noise;
more fragmented clusters.

A larger eps normally produces:

larger neighborhoods;
more core samples;
less noise;
potentially merged clusters.

eps should be selected through a combination of evidence rather than one score.

8. Interpretation of min_samples

The model uses:

min_samples = 5

This defines the minimum neighborhood size required for an observation to become a core sample.

A larger value:

requires stronger local density;
may reduce weak clusters;
may increase noise;
can improve resistance to isolated points.

A smaller value:

allows clusters to form more easily;
may reduce noise;
may produce unstable or weak density regions.
9. K-Distance Plot

The k-distance plot displays observations sorted by their distance to the min_samplesth nearest neighbor.

A visible bend or elbow can provide a starting point for choosing eps.

Interpretation:

values below the bend may represent dense observations;
values above the bend may represent sparse or unusual observations.

The bend is not always clear, especially in high-dimensional data.

Therefore, the chart should be combined with parameter testing and domain interpretation.

10. Cluster Count

DBSCAN determines the number of clusters from density connectivity.

The cluster count excludes noise.

A very large cluster count may suggest:

eps is too small;
density regions are fragmented;
data contain many local groups.

Only one cluster may suggest:

eps is too large;
natural groups are being merged;
the dataset has one dominant dense region.

Zero clusters means all observations were classified as noise.

11. Cluster Sizes

The file:

outputs/metrics/cluster_sizes.csv

shows the number of observations assigned to each cluster and to noise.

Very small clusters may represent:

rare subgroups;
unstable density regions;
local anomalies;
meaningful specialized segments.

Cluster usefulness cannot be judged from size alone.

12. Cluster Profiles

The file:

outputs/metrics/cluster_profiles.csv

contains average original feature values for every cluster and the noise group.

These profiles can be used to:

compare clusters;
identify distinguishing characteristics;
assign business-friendly descriptions;
investigate noise observations.

Cluster labels should be derived from measurable profile differences.

13. Core-Sample File

The file:

outputs/metrics/core_samples.csv

contains observations DBSCAN identified as core samples.

Core samples are useful for:

explaining the dense centre of a cluster;
comparing stable cluster representatives;
approximate assignment of new observations;
understanding density structure.

Core samples are not centroids.

DBSCAN does not calculate cluster centres.

14. Silhouette Score

Silhouette score compares:

cohesion within the assigned cluster;
separation from the nearest alternative cluster.

The range is approximately:

Score	Interpretation
Near 1	Strongly separated
Around 0	Overlapping or boundary observations
Below 0	Potentially questionable assignment

This project excludes noise observations when calculating the silhouette score.

That choice measures the quality of the discovered clusters but does not directly reward or penalize the amount of noise.

A solution with a high silhouette score and excessive noise may not be useful.

15. Davies–Bouldin Score

The Davies–Bouldin score evaluates average similarity between each cluster and its most similar alternative cluster.

Lower scores are generally better.

The score is most useful when comparing different parameter settings on the same standardized dataset.

It should not be interpreted as a percentage.

16. Calinski–Harabasz Score

The Calinski–Harabasz score compares:

between-cluster dispersion;
within-cluster dispersion.

Higher values are generally preferred when comparing solutions on the same dataset.

There is no universal threshold representing good performance.

17. PCA Cluster Plot

The PCA plot projects the standardized feature space into two dimensions.

It visualizes:

core samples;
border observations;
noise;
approximate cluster separation.

However, PCA removes information.

Clusters that overlap in two dimensions may remain separable in the complete feature space.

The DBSCAN model itself uses all standardized features.

18. Parameter Comparison

The file:

outputs/metrics/parameter_comparison.csv

compares combinations of:

eps;
min_samples.

Each combination records:

number of clusters;
number of noise observations;
noise percentage;
silhouette score excluding noise.

A useful parameter setting should balance:

meaningful cluster count;
reasonable noise level;
cluster separation;
profile interpretability;
stability;
business usefulness.

The highest silhouette score alone should not determine the final settings.

19. Parameter Heatmap

The parameter heatmap shows how the number of discovered clusters changes across eps and min_samples.

Stable regions of the heatmap may indicate parameter combinations producing similar structure.

A solution that changes dramatically after a tiny parameter adjustment may be unstable.

20. Approximate New-Observation Assignment

Standard DBSCAN does not provide native prediction for new observations.

The provided prediction script uses an approximation:

scale the new observation;
locate the closest trained core sample;
compare the distance with eps;
assign the core sample’s cluster when sufficiently close;
otherwise classify it as noise.

This method does not:

expand existing clusters;
create new clusters;
retrain DBSCAN;
guarantee the same result as retraining with the new observation.

It is a practical deployment approximation only.

21. Distance to the Nearest Core Sample

The prediction output reports the distance to the nearest learned core sample.

If:

distance <= eps

the new observation is assigned approximately to that core sample’s cluster.

If:

distance > eps

the observation is labelled as noise.

A small distance suggests stronger local similarity.

A distance close to eps indicates a weaker boundary-level assignment.

22. Comparison With K-Means
Aspect	DBSCAN	K-Means
Learning type	Unsupervised	Unsupervised
Cluster number required	No	Yes
Cluster mechanism	Density connectivity	Nearest centroid
Noise detection	Yes	No
Irregular clusters	Stronger	Often weaker
Cluster centres	No	Yes
New prediction	No native support	Native support
Parameter sensitivity	eps, min_samples	k, initialization
Outlier handling	Labels noise	Outliers affect centroids
Different densities	Often challenging	Often challenging

DBSCAN is valuable when density and noise matter.

K-Means is useful when compact centroid-based groups are expected.

23. High-Dimensional Limitations

The Wine dataset contains multiple numerical dimensions.

In higher-dimensional spaces, distance values may become less distinctive.

This can make density estimation difficult.

Possible improvements include:

PCA before DBSCAN;
feature selection;
removal of redundant features;
alternative distance metrics;
HDBSCAN;
domain-specific feature engineering.
24. Business Interpretation

DBSCAN can support applications such as:

identifying dense customer groups;
detecting rare operational behaviour;
separating normal and unusual transactions;
finding geographic hotspots;
identifying product-profile segments;
discovering irregular process patterns.

Noise should not automatically be interpreted as fraud, failure, or error.

Noise means only that the observation did not belong to a sufficiently dense region under the selected parameters.

25. Strengths

DBSCAN provides:

automatic cluster-count discovery;
irregular-shape detection;
explicit noise identification;
core and border distinction;
resistance to isolated observations;
flexible density-based grouping.
26. Limitations

The main limitations are:

sensitivity to eps;
sensitivity to min_samples;
difficulty with varying cluster densities;
scaling dependence;
high-dimensional distance problems;
no native prediction for unseen records;
possible large memory use on some datasets;
parameter instability.
27. Recommended Improvements

Future improvements should include:

Automatic knee-point detection.
PCA before DBSCAN.
Feature-selection experiments.
Alternative distance metrics.
Parameter-stability analysis.
Comparison with HDBSCAN.
Comparison with OPTICS.
Comparison with K-Means.
Noise-profile analysis.
Real customer-segmentation data.
FastAPI approximate-assignment endpoint.
Automated unit tests.
Data-drift checks.
Interactive cluster visualization.
28. Final Conclusion

DBSCAN provides a density-based alternative to centroid-based clustering.

Its main strengths are:

discovering clusters without specifying their number;
detecting irregular groups;
identifying noise;
separating core and border observations.

Its principal weaknesses are:

strong parameter sensitivity;
difficulty with varying densities;
high-dimensional distance challenges;
lack of native prediction for new observations.

In this repository, DBSCAN demonstrates:

density-based clustering;
core, border, and noise concepts;
k-distance parameter analysis;
parameter sensitivity;
non-noise clustering metrics;
PCA visualization;
approximate assignment of new observations.