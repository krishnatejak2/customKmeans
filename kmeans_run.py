import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import kmeans101
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
from matplotlib.image import imread
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score

# %matplotlib inline
sns.set_context("notebook")
plt.style.use("fivethirtyeight")
from warnings import filterwarnings

filterwarnings("ignore")

data, labels = load_iris(return_X_y=True)
print(len(labels))

df = data
# km = kmeans101.customKmeans(4)
# x = km.fit(data.data)


# Standardize the data
X_std = StandardScaler().fit_transform(df)

# Run local implementation of kmeans
km = kmeans101.customKmeans(n_clusters=2, max_iter=100)

km_2 = KMeans(n_clusters=2, max_iter=100)
km.fit(X_std)
centroids = km.centroids
print(km.labels[:5])
km_2.fit(X_std)
print(km_2.labels_[:5])
# Plot the clustered data
fig, ax = plt.subplots(figsize=(6, 6))
plt.scatter(
    X_std[km.labels == 0, 0], X_std[km.labels == 0, 1], c="green", label="cluster 1"
)
plt.scatter(
    X_std[km.labels == 1, 0], X_std[km.labels == 1, 1], c="blue", label="cluster 2"
)
plt.scatter(
    centroids[:, 0], centroids[:, 1], marker="*", s=300, c="r", label="centroid"
)
plt.legend()
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.xlabel("Eruption time in mins")
plt.ylabel("Waiting time to next eruption")
plt.title("Visualization of clustered data", fontweight="bold")
ax.set_aspect("equal")
plt.savefig("./fig1.png")
