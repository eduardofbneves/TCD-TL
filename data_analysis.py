# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

"""
    Column1: Device ID
    Column2: accelerometer x
    Column3: accelerometer y
    Column4: accelerometer z
    Column5: gyroscope x
    Column6: gyroscope y
    Column7: gyroscope z
    Column8: magnetometer x
    Column9: magnetometer y
    Column10: magnetometer z
    Column11: Timestamp
    Column12: Activity Label (16 atividades)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
#import utils


loc = []

for i in range (0,14):
    loc.append("FORTH_TRACE_DATASET-master/part" + str(i) + "/part" + str(i) + "dev2.csv")

df = pd.read_csv(loc[0], sep=',', header=None)
array = df.to_numpy()

# modules of the vectors
t_acc = np.sqrt(np.add(np.square(array[:,[1]]),
                       np.square(array[:,[2]]),
                       np.square(array[:,[3]])))

t_gyr = np.sqrt(np.add(np.square(array[:,[4]]),
                       np.square(array[:,[5]]),
                       np.square(array[:,[6]])))

t_mag = np.sqrt(np.add(np.square(array[:,[7]]),
                       np.square(array[:,[8]]),
                       np.square(array[:,[9]])))


# act1 = t_acc[array[:,-1]==5]

activities = [] # nao sei se util
box_acc = [] # TODO ver se dá para fazer isto com np.array
box_gyr = []
box_mag = []
for i in range(1,17):
    activities.append(array[:,-1]==i)
    '''
    box_acc.append(t_acc[act]) # usar list porque colunas tem tamanhos diferentes
    box_gyr.append(t_gyr[act])
    box_mag.append(t_mag[act])
    #box_acc = [box_acc, t_acc[act]]
    '''
    
activities = np.array(activities).transpose()

#box_acc = np.array(box_acc)
plt.figure()
fig, axs = plt.subplots(3)
axs[0].boxplot(np.multiply(t_acc, activities))
axs[1].boxplot(np.multiply(t_gyr, activities))
axs[2].boxplot(np.ma.masked_equal(np.multiply(t_mag, activities), 0))
#axs[2].boxplot(np.ma.masked_equal(np.multiply(t_mag, activities), 0))


# desvio e outliers para cada k = 3, 3.5, 4
d = []
outliersk = [] 

# for c, d in zip(a, b) itera alternadamente cada lista no mesmo loop
for box in ([box_acc, box_gyr, box_mag]):
    for i in range(0,activities.size):
        # verificar outliers segundo o boxplot
        q1 = np.quantile(box[:][i], 0.25)
        q3 = np.quantile(box[:][i], 0.75)
        med = np.median(box[:][i])
        
        iqr = q3-q1
        upper_bound = q3+(1.5*iqr)
        lower_bound = q1-(1.5*iqr)
        
        # tirar os valores do outliers
        outliers = box[:][i][(box[:][i] <= lower_bound) | (box[:][i] >= upper_bound)]
        out_bool = (box[:][i] <= lower_bound) | (box[:][i] >= upper_bound)
        #print('The following are the outliers in the boxplot:{}'.format(outliers))
        box[:][i] = box[:][i][(box[:][i] >= lower_bound) & (box[:][i] <= upper_bound)]
        
        # unique, counts = np.unique(out_bool, return_counts=True)
        counts = out_bool.sum()
        d.append((counts/out_bool.size)*100) # TODO se a coluna nao tiver desvios
        
        # teste z-score
        zscore = stats.zscore(box[:][i], axis=0, ddof=0, nan_policy='propagate')
        outliersk.append(box[:][i][(zscore <= -3) | (zscore >= 3)])
        #print('The following are the outliers from the z-score test: {}'.format(outliersk[:][i]))

centroids = get_centroids()
plt.figure()
plt.boxplot(box, outliersk, 'gD')
plt.title("k=3")



