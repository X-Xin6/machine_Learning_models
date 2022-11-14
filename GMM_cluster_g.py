# -*- coding: utf-8 -*-

import math
import copy
from collections import defaultdict
import pandas as pd
import numpy as np
import itertools
import random

class GEM:
    def __init__(self, maxstep=10000, epsilon=1e-3, K=3):
        self.maxstep = maxstep
        self.epsilon = epsilon
        self.K = K

        self.alpha = None
        self.mu = None
        self.sigma = None
        self.gamma_all_final = None

        self.D = None
        self.N = None

    def init_param(self, data):

        self.D = data.shape[1]
        self.N = data.shape[0]
        self.get_init_param(data)
        return

    def get_init_param(self, data):
        clusters = defaultdict(list)
        for index, label in enumerate(self.Gaussions(data,3)):
            clusters[label].append(index)
        mu = []
        alpha = []
        sigma = []
        for indexs in clusters.values():
            partial_data = data[indexs]
            mu.append(partial_data.mean(axis=0))
            alpha.append(len(indexs) / self.N)
            sigma.append(np.cov(partial_data.T))
        self.mu = np.array(mu)
        self.alpha = np.array(alpha)
        self.sigma = np.array(sigma)
        return


    def _phi(self, y, mu, sigma):
        s1 = 1.0 / math.sqrt(np.linalg.det(sigma))
        s2 = np.linalg.inv(sigma)  # D*D
        delta = np.array([y - mu])  # 1*D
        return s1 * math.exp(-1.0 / 2 * delta @ s2 @ delta.T)

    def fit(self, data):
        self.init_param(data)
        step = 0
        gamma_all_arr = None
        while step < self.maxstep:
            step += 1
            old_alpha = copy.copy(self.alpha)
            gamma_all = []
            for j in range(self.N):
                gamma_j = []

                for k in range(self.K):
                    gamma_j.append(self.alpha[k] * self._phi(data[j], self.mu[k], self.sigma[k]))

                s = sum(gamma_j)
                gamma_j = [item/s for item in gamma_j]
                gamma_all.append(gamma_j)

            gamma_all_arr = np.array(gamma_all)
            for k in range(self.K):
                gamma_k = gamma_all_arr[:, k]
                SUM = np.sum(gamma_k)
                self.alpha[k] = SUM / self.N
                new_mu = sum([gamma * y for gamma, y in zip(gamma_k, data)]) / SUM
                self.mu[k] = new_mu
                delta_ = data - new_mu
                self.sigma[k] = sum([gamma * (np.outer(np.transpose([delta]), delta)) for gamma, delta in
                                     zip(gamma_k, delta_)]) / SUM
            alpha_delta = self.alpha - old_alpha
            if np.linalg.norm(alpha_delta, 1) < self.epsilon:
                break
        self.gamma_all_final = gamma_all_arr
        return

    def predict(self):
        pre_label=[]
        for j in range(self.N):
            max_ind = np.argmax(self.gamma_all_final[j])
            pre_label.append(max_ind)
        return pre_label
    def Gaussions(self,x, k):
        x = x.astype(float)
        n = x.shape[0]
        ctrs = x[np.random.permutation(x.shape[0])[:k]]
        iter_ctrs = [ctrs]
        idx = np.ones(n)
        x_square = np.expand_dims(np.sum(np.multiply(x, x), axis=1), 1)
        while True:
            distance = -2 * np.matmul(x, ctrs.T)
            distance += x_square
            distance += np.expand_dims(np.sum(ctrs * ctrs, axis=1), 0)
            new_idx = distance.argmin(axis=1)
            if (new_idx == idx).all():break
            idx = new_idx
            ctrs = np.zeros(ctrs.shape)
            for i in range(k):ctrs[i] = np.average(x[idx == i], axis=0)
            iter_ctrs.append(ctrs)
        iter_ctrs = np.array(iter_ctrs)
        return idx


def importdata(datafile):

    data = pd.read_table(datafile,sep=',',header=None)
    data.columns=['sepal_len','sepal_wid','petal_len','petal_wid','label']
    label_dict={"Iris-setosa":0,"Iris-versicolor":1,"Iris-virginica":2}
    data['label']=data['label'].map(label_dict)
    label=data['label'].tolist()
    data=data.iloc[:,:-1].values
    return data,label


if __name__ == '__main__':
    # 文件路径
    data = np.genfromtxt('Data.tsv', delimiter='\t')
    gem = GEM(K=3)
    gem.fit(data)
    y_pre = gem.predict()
    print("GMMpre：\n",y_pre)
    pd.DataFrame(y_pre).to_csv('gmm_output.tsv', index=False,sep='\t',header=None)

    
