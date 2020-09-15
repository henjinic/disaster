import numpy as np
from maps import Maps
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import MinMaxScaler

N_CLUSTERS = 4


def main():
    maps = Maps()

    samples = maps.get_clusterset()

    kmeans = MiniBatchKMeans(n_clusters=N_CLUSTERS, max_iter=100, batch_size=100, random_state=0)

    kmeans.fit(samples)

    print(kmeans.cluster_centers_)
    print(np.unique(kmeans.labels_, return_counts=True))

if __name__ == '__main__':
    main()
