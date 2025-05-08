# Clustering: Approaches for clustering, distance metrics, K-means clustering, expectation maximization, hierarchical clustering, performance evaluation metrics, validation methods. 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.spatial.distance import cdist

# Create synthetic dataset with 3 clusters for demonstration
X, _ = make_blobs(n_samples=300, centers=3, random_state=42)

# Visualize the dataset
plt.scatter(X[:, 0], X[:, 1], c='blue', marker='o', edgecolors='k')
plt.title('Generated Data with 3 Clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# 1. K-means Clustering
# K-means with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

# Plot K-means results
plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', marker='o', edgecolors='k')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', s=100, label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# Performance evaluation for K-means
silhouette_kmeans = silhouette_score(X, kmeans_labels)
davies_bouldin_kmeans = davies_bouldin_score(X, kmeans_labels)
print(f"K-means Silhouette Score: {silhouette_kmeans:.2f}")
print(f"K-means Davies-Bouldin Index: {davies_bouldin_kmeans:.2f}")

# 2. Expectation Maximization (Gaussian Mixture Models)
# Fit a Gaussian Mixture Model with 3 components
gmm = GaussianMixture(n_components=3, random_state=42)
gmm_labels = gmm.fit_predict(X)

# Plot GMM results
plt.scatter(X[:, 0], X[:, 1], c=gmm_labels, cmap='viridis', marker='o', edgecolors='k')
plt.title('Gaussian Mixture Model (Expectation Maximization)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Performance evaluation for GMM
silhouette_gmm = silhouette_score(X, gmm_labels)
davies_bouldin_gmm = davies_bouldin_score(X, gmm_labels)
print(f"GMM Silhouette Score: {silhouette_gmm:.2f}")
print(f"GMM Davies-Bouldin Index: {davies_bouldin_gmm:.2f}")

# 3. Hierarchical Clustering (Agglomerative Clustering)
# Perform Agglomerative Clustering with 3 clusters
agg_clust = AgglomerativeClustering(n_clusters=3)
agg_clust_labels = agg_clust.fit_predict(X)

# Plot Hierarchical Clustering results
plt.scatter(X[:, 0], X[:, 1], c=agg_clust_labels, cmap='viridis', marker='o', edgecolors='k')
plt.title('Agglomerative Clustering (Hierarchical)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Performance evaluation for Hierarchical Clustering
silhouette_agg = silhouette_score(X, agg_clust_labels)
davies_bouldin_agg = davies_bouldin_score(X, agg_clust_labels)
print(f"Hierarchical Clustering Silhouette Score: {silhouette_agg:.2f}")
print(f"Hierarchical Clustering Davies-Bouldin Index: {davies_bouldin_agg:.2f}")

# 4. Clustering Validation Methods
# Silhouette score: Measures how similar an object is to its own cluster compared to other clusters.
# Davies-Bouldin index: Measures the average similarity ratio of each cluster with its most similar one.
# A lower Davies-Bouldin index is better.

# 5. Distance Metrics for Clustering
# Calculate the distance between points to use with clustering algorithms
distances = cdist(X, X, metric='euclidean')  # Euclidean distance matrix

# Plot the distance matrix (optional)
plt.imshow(distances, cmap='viridis')
plt.colorbar()
plt.title('Distance Matrix (Euclidean)')
plt.show()
