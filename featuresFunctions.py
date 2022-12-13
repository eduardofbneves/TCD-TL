import numpy as np
from numpy.linalg import norm
from scipy import fft, stats
    
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

def skewness(array, window):
    skew=[]
    for i in range(round(array.shape[0]/window)):
        i*=window
        skew.append(np.sqrt(np.mean(array[i:(i+window)]**2, axis=0)))
    return np.array(skew)

def kurtosis(array, window):
    kurt=[]
    for i in range(round(array.shape[0]/window)):
        i*=window
        kurt.append(stats.kurtosis(array, axis=0))
    return np.array(kurt)

def irange(array, window):
    ran=[]
    for i in range(round(array.shape[0]/window)):
        i*=window
        ran.append(np.quantile(array[i:(i+window)], 0.75, axis=0) - \
            np.quantile(array[i:(i+window)], 0.25, axis=0))
    return np.array(ran)

def zero_crossing_rate(array, window):
    zcr=[]
    for i in range(round(array.shape[0]/window)):
        i*=window
        zcr.append((((array[i:(i+window)] * array[i:(i+window)]) < 0).sum())/window)
    return np.array(zcr).reshape([-1, 1])


def mean_crossing_rate(array, window):
    mcr=[]
    for i in range(round(array.shape[0]/window)):
        i*=window
        mcr.append(zero_crossing_rate(np.array(array[i:(i+window)]) - np.mean(array[i:(i+window)]), window))
    return np.array(mcr)[:,:,0]

def spectral_entropy(array, window):
    se = []
    for i in range(round(array.shape[0]/window)):
        i*=window
        se.append(fft.fft(array[i:(i+window)]))
    return np.array(se).reshape([-1, 1])

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
    return np.array(sma, dtype=object).reshape(-1,1)

def eva(array, window):
    eva=[]
    for i in range(round(array.shape[0]/window)):
        i*=window
        cov = (np.cov(array[i:(i+window)]))
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        eva.append(eigenvectors)
    return np.array(eva).reshape([-1,1])

def cagh(head1, head2, grav, window):
    cagh = []
    for i in range(round(head1.shape[0]/window)):
        i*=window
        euc = np.sqrt(np.add(np.square(head1[i:(i+window)]),
                            np.square(head2[i:(i+window)])))
        cagh.append(np.cov(euc, grav[i:(i+window)])[0,1]) # TODO cov e 2x2
    return np.array(cagh).reshape([-1,1])
  
def df_energy_aae_are(array, window):
    domf = []
    ener = []
    aae = []
    are = []
    for i in range(round(array.shape[0]/window)):
        fourier = np.square(fft.fft(array[i:(i+window)]))
        domf.append(np.max(fourier))
        ener.append(np.sum(fourier)/array.shape[0])
        aae.append(np.mean(ener[0:2]))
        are.append(np.mean(ener[3:5]))
    return np.array(domf).reshape([-1, 1]), np.array(ener).reshape([-1, 1]), \
        np.array(aae).reshape([-1, 1]), np.array(are).reshape([-1, 1])
    
