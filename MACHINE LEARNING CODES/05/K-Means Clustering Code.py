import numpy as np

class KMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters

    def fit(self, X, initial_centroids):
        self.centroids = initial_centroids
        for _ in range(self.max_iters):
            clusters = [[] for _ in range(self.k)]
            for x in X:
                distances = [np.linalg.norm(x - c) for c in self.centroids]
                clusters[np.argmin(distances)].append(x)
            new_centroids = [np.mean(cluster, axis=0) for cluster in clusters]
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = np.array(new_centroids)
        self.clusters = clusters

# Sample data
data = np.array([[2, 10], [2, 5], [8, 4], [5, 8], [7, 5], [6, 4], [1, 2], [4, 9]])

# Define initial centroids A1, B1, C1
initial_centroids = np.array([[2, 10], [5, 8], [1, 2]])

# Initialize and fit KMeans
kmeans = KMeans()
kmeans.fit(data, initial_centroids)

# Print clusters
for i, cluster in enumerate(kmeans.clusters):
    print(f'Cluster {i+1} (Center: {"ABC"[i]}1): {cluster}')

# Print centroids
print('Centroids:')
for i, centroid in enumerate(kmeans.centroids):
    print(f'{["A", "B", "C"][i]}1:', centroid)
