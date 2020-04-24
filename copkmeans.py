import random 


def cop_kmeans(dataset, k, ml=[], cl=[], initialization="kmpp", max_iter=300, tol=1e-4):
    # find the transitive closure and new graphs for ml and cl constraints
    ml, cl = transitive_closure(ml, cl, len(dataset))
    # find info
    ml_info = get_ml_info(ml, dataset)
    # achieve tolerance
    tol = tolerance(tol, dataset)

    # initialize k centeres through 'kmeans++' initialization - for better centroid stability
    centers = initialize_centers(dataset, k, initialization)

    for _ in range(max_iter):
        # mark all the datapoints to '-1' cluster and fill the cluster labels later
        clusters_ = [-1] * len(dataset)
        # for each data point
        for i, d in enumerate(dataset):
            # find the closest centroid index
            indices, _ = closest_clusters(centers, d)
            counter = 0
            # if cluster is not assigned for a data point then :
            if clusters_[i] == -1:
                found_cluster = False
                while (not found_cluster) and counter < len(indices):
                    index = indices[counter]
                    # if constriants are not violated, then assign same clutser to all points of ml
                    if not violate_constraints(i, index, clusters_, ml, cl):
                        found_cluster = True
                        clusters_[i] = index
                        for j in ml[i]:
                            clusters_[j] = index
                    counter += 1

                if not found_cluster:
                    return None, None
        # compute new cluster centers
        clusters_, centers_ = compute_centers(clusters_, dataset, k, ml_info)
        shift = sum(l2_distance(centers[i], centers_[i]) for i in range(k))
        # break loop if shift < tol
        if shift <= tol:
            break

        centers = centers_

    return clusters_, centers_


# simple euclidian distance summed over for all the points
def l2_distance(point1, point2):
    return sum([(float(i) - float(j)) ** 2 for (i, j) in zip(point1, point2)])


# tolerance as in kmeans algorithm : algorithm stops as the difference between new clusters is minimal
def tolerance(tol, dataset):
    import numpy as np

    dim = len(dataset[0])
    variances = np.var(dataset, axis=0)
    return tol * np.sum(variances) / dim


# find the closest clusters between all centroids and data points : returns a sorted index
def closest_clusters(centers, datapoint):
    distances = [l2_distance(center, datapoint) for center in centers]
    return sorted(range(len(distances)), key=lambda x: distances[x]), distances


# initialize centroids - kmeans++ algorithm
def initialize_centers(dataset, k, method):
    # if method = random, then randoomly k points are picked up as centroids
    if method == "random":
        ids = list(range(len(dataset)))
        random.shuffle(ids)
        return [dataset[i] for i in ids[:k]]
    # if kmeans++ is chosen, then :
    elif method == "kmpp":
        # each datapoint is given equal chance
        chances = [1] * len(dataset)
        centers = []

        for _ in range(k):
            chances = [x / sum(chances) for x in chances]
            r = random.random()
            acc = 0.0
            for index, chance in enumerate(chances):
                if acc + chance >= r:
                    break
                acc += chance
            # pick an index off the dataset if acc+chance >= r
            centers.append(dataset[index])

            # for index, point in enumerate(dataset):
            #     cids, distances = closest_clusters(centers, point)
            #     chances[index] = distances[cids[0]]

        return centers


def violate_constraints(data_index, cluster_index, clusters, ml, cl):

    for i in ml[data_index]:
        if clusters[i] != -1 and clusters[i] != cluster_index:
            return True

    for i in cl[data_index]:
        if clusters[i] == cluster_index:
            return True

    return False


def compute_centers(clusters, dataset, k, ml_info):
    # inorder to compute new clusters, ml info is always taken into considerations
    cluster_ids = set(clusters)
    k_new = len(cluster_ids)
    id_map = dict(zip(cluster_ids, range(k_new)))
    clusters = [id_map[x] for x in clusters]

    dim = len(dataset[0])
    centers = [[0.0] * dim for i in range(k)]

    counts = [0] * k_new
    for j, c in enumerate(clusters):
        for i in range(dim):
            centers[c][i] += dataset[j][i]
        counts[c] += 1

    for j in range(k_new):
        for i in range(dim):
            centers[j][i] = centers[j][i] / float(counts[j])

    if k_new < k:
        ml_groups, ml_scores, ml_centroids = ml_info
        current_scores = [
            sum(l2_distance(centers[clusters[i]], dataset[i]) for i in group)
            for group in ml_groups
        ]
        group_ids = sorted(
            range(len(ml_groups)),
            key=lambda x: current_scores[x] - ml_scores[x],
            reverse=True,
        )

        for j in range(k - k_new):
            gid = group_ids[j]
            cid = k_new + j
            centers[cid] = ml_centroids[gid]
            for i in ml_groups[gid]:
                clusters[i] = cid

    return clusters, centers


def get_ml_info(ml, dataset):
    # set all flags list to 1
    flags = [True] * len(dataset)
    groups = []
    # for each datapoint, find its group from ml_graph or if there is no graph, then the point itself
    for i in range(len(dataset)):
        if not flags[i]:
            continue
        group = list(ml[i] | {i})
        # append this grouplist to original 'groups' list
        groups.append(group)
        # set flag = false for each element in the selected group
        for j in group:
            flags[j] = False
    # dimension - no of columns
    dim = len(dataset[0])
    # scores - 0 for all groups
    scores = [0.0] * len(groups)
    # centroids - 0 for all dimentions and for all the groups
    centroids = [[0.0] * dim for i in range(len(groups))]

    # for each group : calculate centroid - mean
    for j, group in enumerate(groups):
        for d in range(dim):
            for i in group:
                centroids[j][d] += dataset[i][d]
            centroids[j][d] /= float(len(group))
    # for each group, calculate l2 score between the centroid and dataset
    scores = [
        sum(l2_distance(centroids[j], dataset[i]) for i in groups[j])
        for j in range(len(groups))
    ]

    return groups, scores, centroids


def transitive_closure(ml, cl, n):
    # initialize ml and cl as dict for each data point in the data
    # it has a set under each data point 'key' in the dict
    ml_graph = dict()
    cl_graph = dict()
    for i in range(n):
        ml_graph[i] = set()
        cl_graph[i] = set()

    # when a point is reated to other point, the reverse is also possible
    def add_both(d, i, j):
        d[i].add(j)
        d[j].add(i)

    # for all links in ml, both ways relation is added
    for (i, j) in ml:
        add_both(ml_graph, i, j)
    # DFS algorithm to denote each visit to the node.
    # this ensures all the transitive links from each node
    def dfs(i, graph, visited, component):
        visited[i] = True
        for j in graph[i]:
            if not visited[j]:
                dfs(j, graph, visited, component)
        component.append(i)

    visited = [False] * n
    for i in range(n):
        if not visited[i]:
            component = []
            dfs(i, ml_graph, visited, component)
            for x1 in component:
                for x2 in component:
                    if x1 != x2:
                        # adds all the data points x1 must-link with - transitive conditions will be fullfilled
                        ml_graph[x1].add(x2)
    # for each cannot link criterion, transitive closure should also be checked - by the paper
    for (i, j) in cl:
        add_both(cl_graph, i, j)
        # for cl of (i,j), 'i' should not link with any other point through 'j'
        for y in ml_graph[j]:
            add_both(cl_graph, i, y)
        # for cl of (i,j), 'j' should not link with any other point through 'i'
        for x in ml_graph[i]:
            add_both(cl_graph, x, j)
            # one more loop to close
            for y in ml_graph[j]:
                add_both(cl_graph, x, y)

    for i in ml_graph:
        for j in ml_graph[i]:
            if j != i and j in cl_graph[i]:
                # raise exception if
                raise Exception("inconsistent constraints between %d and %d" % (i, j))

    return ml_graph, cl_graph


# from __future__ import print_function

# from cop_kmeans import cop_kmeans, l2_distance
import argparse


def read_data(datafile):
    import numpy as np

    # read data as a list
    data = []
    with open(datafile, "r") as f:
        for line in f:
            line = line.strip()
            if line != "":
                # split each line of data in the datafile into number of columns
                d = [float(i) for i in line.split()]
                data.append(d)
    # return a numpy array
    return np.array(data)


def read_constraints(consfile):
    # initialize must-link and cannot-link
    ml, cl = [], []
    with open(consfile, "r") as f:
        for line in f:
            line = line.strip()
            if line != "":
                line = line.split()
                # datapoints in the first 2 columns of th file
                constraint = (int(line[0]), int(line[1]))
                # ml or cl relation in the 3rd column of the file
                c = int(line[2])
                if c == 1:
                    ml.append(constraint)
                if c == -1:
                    cl.append(constraint)
    return ml, cl


def run(datafile, consfile, k, n_rep, max_iter, tolerance):
    import pandas as pd
    import numpy as np

    # read data file as numpy array
    data = read_data(datafile)
    # read constraints file and classify ml and cl
    ml, cl = read_constraints(consfile)
    # Initialize clusters and Scores
    best_clusters = None
    best_score = None
    # n_rep - corresponds to CrossValidation(CV - default 10)
    for _ in range(n_rep):
        # calls cop-kmeans fucntion declared above
        clusters, centers = cop_kmeans(
            data, k, ml, cl, max_iter=max_iter, tol=tolerance
        )
        if clusters is not None and centers is not None:
            # Sum of squares from all the points to their respective clusters
            score = sum(
                l2_distance(data[j], centers[clusters[j]]) for j in range(len(data))
            )
            if best_score is None or score < best_score:
                # update if the new score is better than the last score
                best_score = score
                best_clusters = clusters
    # return final clusters
    return best_clusters


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Run COP-Kmeans algorithm")
    # parser.add_argument("dfile", help="data file")
    # parser.add_argument("cfile", help="constraint file")
    # parser.add_argument("k", type=int, help="number of clusters")
    # parser.add_argument("--ofile", help="file to store the output", default=None)
    # parser.add_argument(
    #     "--n_rep", help="number of times to repeat the algorithm", default=10, type=int
    # )
    # parser.add_argument(
    #     "--m_iter",
    #     help="maximum number of iterations of the main loop",
    #     default=300,
    #     type=int,
    # )
    # parser.add_argument(
    #     "--tol", help="tolerance for deciding on convergence", default=1e-4, type=float
    # )
    # args = parser.parse_args()

    # clusters = run("./iris.data.txt", "./iris.constraints.txt", 3, 10, 100, 0.001)
    import sys

    sys.setrecursionlimit(10 ** 6)

    clusters = run("./data.txt", "./constraints_file.csv", 3, 10, 500, 0.0001)
    ofile = "./output.txt"
    if ofile is not None and clusters is not None:
        with open(ofile, "w") as f:
            for cluster in clusters:
                f.write("%d\n" % cluster)

    if not clusters:
        print("No solution was found!")
    else:
        print(" ".join(str(c) for c in clusters))
