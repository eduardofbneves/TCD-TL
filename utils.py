# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 16:53:53 2022

Produced to apply on a vector of data for a boxplot
@author: Eduardo
"""

import numpy as np
from numpy.linalg import norm
from sklearn import datasets, linear_model, metrics
from scipy import fft

# X.sample usado em DataFrame e nao nparrays

def get_centroids(data, clusters):
    if type(data) != np.ndarray:
        data = np.array(data)
    rand_id = np.random.permutation(data.shape[0])
    centroids = data[rand_id[:(clusters)]] # cria 3 centroides para cada coluna
    return centroids

def get_distance(data, centroids, clusters):
    distance = np.zeros((data.shape[0], clusters))
    for k in range(clusters):
        row_norm = data - centroids[k] # TODO reformular isto
        distance[:, k] = np.square(row_norm)
    return distance

def compute_centroids(data, clusters, distance):
    centroids = np.zeros((clusters, 1))
    arr_min = np.argmin(distance, axis=1)
    for k in range(clusters):
        centroids[k, :] = np.mean(data[arr_min == k], axis=0)
    return centroids

def k_means(data, clusters):
    centroids = get_centroids(data, clusters)
    diff = 1
    while diff!=0:
        old_centroids = centroids
        distance = get_distance(data, old_centroids, clusters)
        centroids = compute_centroids(data, clusters, distance)
        diff = np.sum(np.subtract(centroids, old_centroids))
        dist = get_distance(data, centroids, clusters)
        cluster = np.argmin(dist, axis=1)
    return centroids, cluster

def get_outliers(vec):
    q1 = np.quantile(vec, 0.25)
    q3 = np.quantile(vec, 0.75)
    #av = np.average(vec)
    
    iqr = q3-q1
    upper_bound = q3+(1.5*iqr)
    lower_bound = q1-(1.5*iqr)

    outliers = vec[(vec <= lower_bound) | (vec >= upper_bound)]
    out_bool = (vec <= lower_bound) | (vec >= upper_bound)
    
    counts = np.count_nonzero(out_bool==True)
    d = (counts/out_bool.size)*100
    return outliers, d

def inject_outliers(x, d, data, p):
    if (x>d):
        points = round((x-d)*p*0.01)
        median = np.median(data)
        sd = np.std(data)
        s = (np.random.random()*2)-1
        rang = np.ptp(data) # range
        for i in range(points):
            q = np.random.random()*rang
            point = round(np.random.random()*(data.shape[0])-1)
            data[point] = median+s*3*(sd+q)
    return data

def fit_linear(X, Y, n):
    #X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1)
    # TODO e suposto usar a cena de cima
    reg = linear_model.LinearRegression()
    reg.fit(X, Y[:n].reshape(-1,1))
    return reg.coef_
    
def mean(array, window):
    
    mean = []
    
    for i in range(array.shape[0] - window + 1):
        mean.append(np.meandata[i:(i+window)])

def median(data, window):
    
def cagh(head1, head2, grav):
    euc = np.sqrt(np.add(np.square(head1),
                           np.square(head2)))
    out = np.cov(euc, grav)
    return out
  
def df_energy(array):
    domf = []
    ener = []
    for i in range(array):
        fourier = np.square(fft.fft(i))
        domf.append(np.max(fourier))
        ener.append(np.sum(fourier)/array.shape[0])
    aae = np.mean(ener[0:2])
    are = np.mean(ener[3:5])
    return domf, ener, aae, are
    




'''
class K_means:
    

    def __init__(self, n_clusters, max_iter=100, random_state=123):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

    def initializ_centroids(self, X):
        np.random.RandomState(self.random_state)
        random_idx = np.random.permutation(X.shape[0])
        centroids = X[random_idx[:self.n_clusters]]
        return centroids

    def compute_centroids(self, X, labels):
        centroids = np.zeros((self.n_clusters, X.1))
        for k in range(self.n_clusters):
            centroids[k, :] = np.mean(X[labels == k, :], axis=0)
        return centroids

    def compute_distance(self, X, centroids):
        distance = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            row_norm = norm(X - centroids[k, :], axis=1)
            distance[:, k] = np.square(row_norm)
        return distance

    def find_closest_cluster(self, distance):
        return np.argmin(distance, axis=1)

    def compute_sse(self, X, labels, centroids):
        distance = np.zeros(X.shape[0])
        for k in range(self.n_clusters):
            distance[labels == k] = norm(X[labels == k] - centroids[k], axis=1)
        return np.sum(np.square(distance))
    
    def fit(self, X):
        self.centroids = self.initializ_centroids(X)
        for i in range(self.max_iter):
            old_centroids = self.centroids
            distance = self.compute_distance(X, old_centroids)
            self.labels = self.find_closest_cluster(distance)
            self.centroids = self.compute_centroids(X, self.labels)
            if np.all(old_centroids == self.centroids):
                break
        self.error = self.compute_sse(X, self.labels, self.centroids)
    
    def predict(self, X):
        distance = self.compute_distance(X, self.centroids)
        return self.find_closest_cluster(distance)
    '''