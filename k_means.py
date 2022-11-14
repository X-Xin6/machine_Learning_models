import numpy as np
import pandas as pd

data = np.genfromtxt('Data.tsv', delimiter='\t')


class KMeans:

    def __init__(self, k, max_iter=1000, dist_type='l2'):
        self.cluster_num = k
        self.max_iter = max_iter
        self.dist_type = dist_type
        self.centers = np.array([[1.03800476, 0.09821729, 1.0469454, 1.58046376],
                                 [0.18982966, -1.97355361, 0.70592084, 0.3957741],
                                 [1.2803405, 0.09821729, 0.76275827, 1.44883158]])

        self.dists = None
        self.labels = None

    def fit(self, samples):
        for _iter in range(self.max_iter):
            self.update_dists(samples)
            centers = self.update_centers(samples)
            if (centers == self.centers).all():
                print('Current iter:', _iter)
                break
            else:
                self.centers = centers

    def update_dists(self, samples):
        labels = np.empty((samples.shape[0]))
        dists = np.empty((0, self.cluster_num))
        for i, sample in enumerate(samples):
            if self.dist_type == 'l1':
                dist = self.l1_distance(sample)
            elif self.dist_type == 'l2':
                dist = self.l2_distance(sample)
            else:
                raise ValueError('wrong dist_type')
            labels[i] = np.argmin(dist)
            dists = np.vstack((dists, dist))
        self.dists = dists
        self.labels = labels

    def update_centers(self, samples):
        centers = np.empty((0, samples.shape[1]))
        for i in range(self.cluster_num):
            idx = (self.labels == i)
            center_samples = samples[idx]
            if len(center_samples) > 0:
                center = np.mean(center_samples, axis=0)
            else:
                center = self.centers[i]
            centers = np.vstack((centers, center[np.newaxis, :]))
        return centers

    def l1_distance(self, sample):
        return np.sum(np.abs(self.centers - sample), axis=1)

    def l2_distance(self, sample):
        return np.sum(np.square(self.centers - sample), axis=1)


if __name__ == '__main__':
    num_cluster = 3
    kmeans = KMeans(num_cluster)
    kmeans.fit(data)
    print(kmeans.centers)
    pd.DataFrame(kmeans.labels).to_csv('kmeans_output.tsv', index=False,sep='\t',header=None)#保存
