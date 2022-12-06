import numpy as np
from numpy.linalg import norm
from sklearn import datasets, linear_model, metrics
from scipy import fft, stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# X.sample usado em DataFrame e nao nparrays

def zscore(scores, k, axis=0):
    
    a = np.asanyarray(scores)
    if a.size == 0:
        return np.empty(a.shape)
    
    mn = np.mean(a, axis = axis)
    std = np.std(a, axis = axis)
    
    z = (scores - mn) / std
    outliers = scores[(scores < -k) | (scores > k)]
    
    return z, outliers


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
    #for i in range(array.shape[0] - window + 1)
    for i in range(round(array.shape[0]/window)):
        i*=window
        mean.append(np.mean(array[i:(i+window)], axis=0))
    return np.array(mean)

def median(array, window):
    median=[]
    for i in range(round(array.shape[0]/window)):
        i*=window
        median.append(np.median(array[i:(i+window)], axis=0))
    return np.array(median)

def std_var(array, window):
    std=[]
    var=[]
    val = 0
    for i in range(round(array.shape[0]/window)):
        i*=window
        val = np.std(array[i:(i+window)], axis=0)
        std.append(val)
        var.append(np.square(val))
    return np.array(std), np.array(var)


def rms(array, window):
    rms=[]
    for i in range(round(array.shape[0]/window)):
        i*=window
        rms.append(np.sqrt(np.mean(array[i:(i+window)]**2, axis=0)))
    return np.array(rms)

'''
def av_derivatives(array, window):
    av=[]
    for i in range(round(array.shape[0]/window)):
        i*=window
        av.append(array[i:(i+window)], axis=0)
    return np.array(av)
'''

def irange(array, window):
    ran=[]
    for i in range(round(array.shape[0]/window)):
        i*=window
        ran.append(np.quantile(array[i:(i+window)], 0.75, axis=0) - \
            np.quantile(array[i:(i+window)], 0.25, axis=0))
    return np.array(ran)

def mov_intensity(t_acc, window):
    ai = []
    vi = []
    for i in range(round(t_acc.shape[0]/window)):
        i*=window
        ai.append(np.sum(t_acc[i:(i+window)])/window)
    
    for i in range(round(t_acc.shape[0]/window)):
        i*=window
        vi.append(np.sum(t_acc[i:(i+window)]-np.mean(ai))/window)
    return np.array(ai).reshape(-1,1), np.array(vi).reshape(-1,1)

def sma(array, window):
    sma=[]
    array = np.absolute(array)
    for i in range(round(array.shape[0]/window)):
        i*=window
        sma.append(np.sum(array[i:(i+window)], axis = 1))
    return np.array(sma).reshape(-1,1)

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
    return np.array(domf), np.array(ener), aae, are
    

def pca_feature(x, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(x, y, \
        test_size = test_size, random_state=0)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    pca = PCA()

    return pca, X_train, X_test, y_train, y_test


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