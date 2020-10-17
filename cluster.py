import numpy as np
from maps import Maps
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import MinMaxScaler


class Cluster:

    def __init__(self, n_clusters=4):
        self._kmeans = MiniBatchKMeans(n_clusters=n_clusters)

    def fit(self, samples):
        self._kmeans.fit(samples)

        labels = self._kmeans.labels_
        centers = self._kmeans.cluster_centers_

        return labels, centers


def main():
    samples = [[1, 1], [2, 2], [3, 5], [4, 5], [8, 6], [7, 2], [5, 6]]

    cluster = Cluster(n_clusters=4)
    labels, centers = cluster.fit(samples)

    print(labels)
    print(centers)


if __name__ == '__main__':
    main()
