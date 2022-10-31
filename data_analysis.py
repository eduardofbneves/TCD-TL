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

loc = []

for i in range (0,14):
    loc.append("FORTH_TRACE_DATASET-master/part" + str(i) + "/part" + str(i) + "dev2.csv")

df = pd.read_csv(loc[0], sep=',', header=None)
array = df.to_numpy()


t_acc = np.sqrt(np.add(np.square(array[:,1]),
                       np.square(array[:,2]),
                       np.square(array[:,3])))

t_gyr = np.sqrt(np.add(np.square(array[:,4]),
                       np.square(array[:,5]),
                       np.square(array[:,6])))

t_mag = np.sqrt(np.add(np.square(array[:,7]),
                       np.square(array[:,8]),
                       np.square(array[:,9])))


# act1 = t_acc[array[:,-1]==5]
act = (array[:,-1]==i)

activities = np.arange(1, 17) # nao sei se util
box_array = [] # TODO ver se dá para fazer isto com np.array
for i in range(1,17):
    act = (array[:,-1]==i)
    box_array.append(t_acc[act]) # usar list porque colunas tem tamanhos diferentes
    #box_array = [box_array, t_acc[act]]

#box_array = np.array(box_array)
plt.figure()
plt.boxplot(box_array, 0, 'gD')
plt.title("Acelerómetro")


for i in range(0,activities.size):
    q1 = np.quantile(box_array[:][i], 0.25)
    q3 = np.quantile(box_array[:][i], 0.75)
    med = np.median(box_array[:][i])
    
    iqr = q3-q1

    upper_bound = q3+(1.5*iqr)
    lower_bound = q1-(1.5*iqr)
    print(iqr, upper_bound, lower_bound)

    outliers = box_array[:][i][(box_array[:][i] <= lower_bound) | (box_array[:][i] >= upper_bound)]
    print('The following are the outliers in the boxplot:{}'.format(outliers))
    box_array[:][i] = box_array[:][i][(box_array[:][i] >= lower_bound) & (box_array[:][i] <= upper_bound)]
    
plt.figure(figsize=(12, 7))
plt.boxplot(box_array)
plt.show()

'''
fig, ax = plt.subplots(3, 1)
pos = np.arange(len(treatments)) + 1

fig, ax = plt.subplots()
ax.boxplot(t_acc, 0, 'gD') 
fig, ax = plt.subplots()
ax.boxplot(t_gyr)
fig, ax = plt.subplots()
ax.boxplot(t_mag)
'''