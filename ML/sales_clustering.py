
# K-Means and Hierarchical Clustering on Sales Data

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# 1. Load the dataset
df = pd.read_csv("sales_data_sample.csv", encoding="unicode_escape")

print("First 5 rows:")
print(df.head())
print("\nDataset shape:", df.shape)

# 2. Select numerical features for clustering
features = ["QUANTITYORDERED", "PRICEEACH", "SALES"]
X = df[features].copy()

# 3. Handle missing values
X = X.fillna(X.median())

# 4. Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Elbow Method: find a suitable number of clusters
wcss = []

for k in range(1, 11):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker="o")
plt.title("Elbow Method for Optimal Number of Clusters")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.xticks(range(1, 11))
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. Choose K based on the elbow graph
# Start with K=3; change this if your graph suggests another elbow.
optimal_k = 3
print("\nSelected number of clusters:", optimal_k)

# 7. Apply K-Means clustering
kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=10
)

df["KMeans_Cluster"] = kmeans.fit_predict(X_scaled)

print("\nK-Means cluster counts:")
print(df["KMeans_Cluster"].value_counts().sort_index())

# 8. Apply Hierarchical clustering
hierarchical = AgglomerativeClustering(
    n_clusters=optimal_k,
    linkage="ward"
)

df["Hierarchical_Cluster"] = hierarchical.fit_predict(X_scaled)

print("\nHierarchical cluster counts:")
print(df["Hierarchical_Cluster"].value_counts().sort_index())

# 9. Visualize K-Means clusters
plt.figure(figsize=(8, 5))
plt.scatter(
    df["QUANTITYORDERED"],
    df["SALES"],
    c=df["KMeans_Cluster"],
    cmap="viridis",
    s=20
)
plt.title("K-Means Clustering of Sales Data")
plt.xlabel("Quantity Ordered")
plt.ylabel("Sales")
plt.tight_layout()
plt.show()

# 10. Draw a dendrogram using a sample for readability
sample_size = min(100, len(X_scaled))
linked = linkage(X_scaled[:sample_size], method="ward")

plt.figure(figsize=(12, 5))
dendrogram(linked)
plt.title("Hierarchical Clustering Dendrogram (Sample)")
plt.xlabel("Sample Record")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()

# 11. Save results
df.to_csv("sales_clustering_results.csv", index=False)

print("\nClustered data saved to sales_clustering_results.csv")
print("Program executed successfully.")
