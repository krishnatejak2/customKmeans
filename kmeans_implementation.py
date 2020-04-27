from numpy.linalg import norm

import numpy as np
import scipy.stats
import pandas as pd
import seaborn as sns
import random
from matplotlib import cm
import matplotlib.pyplot as plt
from itertools import combinations, product, chain

# %matplotlib inline

from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import (
    KMeans,
    DBSCAN,
    SpectralClustering,
    AgglomerativeClustering,
    Birch,
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import svm


n = 500
mean_1 = [0, 0]
cov_1 = [[10, -50], [-50, 100]]
# x,y = np.random.multivariate_normal(mean_1,cov_1,n).T

mean_2 = [30, 50]
cov_2 = [[10, 70], [70, 100]]
# x_2,y_2 = np.random.multivariate_normal(mean_2,cov_2,n).T
random.seed(123)
x, y = np.append(
    np.random.multivariate_normal(mean_1, cov_1, n).T,
    np.random.multivariate_normal(mean_2, cov_2, n).T,
    axis=1,
)

final_data = pd.DataFrame({"x": x, "y": y})
del x, y
final_data["label"] = np.append(np.repeat(0, n), np.repeat(1, n), axis=0)
final_data["Constrained_label"] = [
    2 if (xx > 0 and yy < 0) else zz
    for (xx, yy, zz) in zip(final_data.x, final_data.y, final_data.label)
]
final_data.reset_index(inplace=True)
sns.scatterplot(final_data.x, final_data.y, hue=final_data.Constrained_label).set_title(
    "2D Random Bimodal Distribution"
)
plt.savefig("./KMeans_ExpectedOutput.png")

final_data_label_array = np.array(final_data.iloc[:, 3:4]).ravel()
final_data_array = np.array(final_data.iloc[:, 1:3])

# print(final_data)


class Kmeans:
    """Implementing Kmeans algorithm."""

    def __init__(self, n_clusters, condition, max_iter=100, random_state=123):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.condition = condition

    def initialize_centroids(self, X, condition):
        print("initializing centroids...")

        np.random.seed(self.random_state)
        x = True
        while x:
            random_idx = np.random.permutation(X.shape[0])
            self.centroids = X[random_idx[: self.n_clusters]]
            check_arr = [
                True
                if (eval("x " + condition[0])) and (eval("y " + condition[1]))
                else False
                for (x, y) in self.centroids
            ]
            if (np.sum(check_arr) == 1) and (check_arr[self.n_clusters - 1] == True):
                x = False
        return self.centroids

    def compute_centroids(self, X, labels, labels_condition, condition):
        centroids = np.zeros((self.n_clusters, X.shape[1]))

        for k in range(self.n_clusters - 1):
            centroids[k, :] = np.mean(self.X_normal[(labels == k), :], axis=0)
        centroids[k + 1, :] = np.mean(self.X_condition, axis=0)

        return centroids

    def compute_distance(self, X, centroids, condition):
        X_normal = X[
            ~(
                (eval("X[:,0] " + self.condition[0]))
                & (eval("X[:,1] " + self.condition[1]))
            )
        ]
        distance = np.zeros((X_normal.shape[0], self.n_clusters - 1))
        for k in range(self.n_clusters - 1):
            row_norm = norm(X_normal - centroids[k, :], axis=1)
            distance[:, k] = np.square(row_norm)
        return distance

    def find_closest_cluster(self, distance):
        x = np.argmin(distance, axis=1)
        return x

    def compute_sse(self, X, labels, centroids, condition):
        distance = np.zeros(self.X_normal.shape[0])
        for k in range(self.n_clusters - 1):
            distance[labels == k] = norm(
                self.X_normal[labels == k] - centroids[k], axis=1
            )
        return np.sum(np.square(distance))

    def fit(self, X, labels):
        from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

        self.X_condition = X[
            (eval("X[:,0] " + self.condition[0]))
            & (eval("X[:,1] " + self.condition[1]))
        ]
        self.X_normal = X[
            ~(
                (eval("X[:,0] " + self.condition[0]))
                & (eval("X[:,1] " + self.condition[1]))
            )
        ]
        self.X_condition_labels = labels[
            (eval("X[:,0] " + self.condition[0]))
            & (eval("X[:,1] " + self.condition[1]))
        ]
        self.X_normal_labels = labels[
            ~(
                (eval("X[:,0] " + self.condition[0]))
                & (eval("X[:,1] " + self.condition[1]))
            )
        ]
        self.labels_original = np.concatenate(
            (self.X_condition_labels, self.X_normal_labels)
        )

        self.centroids = self.initialize_centroids(X, self.condition)
        for i in range(self.max_iter):
            self.old_centroids = self.centroids
            distance = self.compute_distance(X, self.old_centroids, self.condition)
            self.labels = self.find_closest_cluster(distance)
            self.labels_condition = [
                self.n_clusters - 1 for i in range(len(self.X_condition))
            ]
            self.centroids = self.compute_centroids(
                X, self.labels, self.labels_condition, self.condition
            )
            if np.all(self.old_centroids == self.centroids):
                break
        # print(self.centroids)
        self.error = self.compute_sse(X, self.labels, self.centroids, self.condition)
        self.result_data = np.concatenate((self.X_condition, self.X_normal))
        self.original_labels = np.concatenate((self.X_condition, self.X_normal))
        self.final_labels = np.concatenate((self.labels_condition, self.labels))
        print(
            "NMI : ",
            normalized_mutual_info_score(self.labels_original, self.final_labels),
        )

    def predict(self, X):
        X_condition = X[
            (eval("X[:,0] " + self.condition[0]))
            & (eval("X[:,1] " + self.condition[1]))
        ]
        X_normal = X[
            ~(
                (eval("X[:,0] " + self.condition[0]))
                & (eval("X[:,1] " + self.condition[1]))
            )
        ]

        distance = self.compute_distance(X_normal, self.old_centroids, self.condition)
        labels = self.find_closest_cluster(distance)
        labels_condition = np.array([self.n_clusters for i in range(len(X_condition))])
        labels_result = np.concatenate((labels, labels_condition))
        data_result = np.concatenate((X_normal, X_condition))
        return (data_result, labels_result)


km = Kmeans(3, ["> 0", "< 0"], max_iter=100, random_state=123)
km.fit(final_data_array, final_data_label_array)
result = km.predict(final_data_array)
# print(km.result_data)
# print(km.centroids)

sns.scatterplot(
    x=km.result_data[:, 0], y=km.result_data[:, 1], hue=km.final_labels, legend=False
).set_title("Output Clusters")
plt.savefig("./KMeans_Output.png")

plt.figure(figsize=(15, 5))
plt.subplot(121)
sns.scatterplot(final_data.x, final_data.y, hue=final_data.Constrained_label).set_title(
    "2D Random Bimodal Distribution"
)
plt.subplot(122)
# sns.scatterplot(data.x, data.y, hue=data.Constrained_label).set_title(
#     "2D Random Bimodal Distribution - Desired Clusters"
# )
# plt.savefig("./ExpectedOutput.png")
sns.scatterplot(
    x=result[0][:, 0], y=result[0][:, 1], hue=result[1], legend=False
).set_title("Predicted Clusters")
plt.savefig("./KMeans_Predict_Output.png")
