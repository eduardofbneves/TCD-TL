# %% [markdown]
# 
# - Column1: Device ID
# - Column2: accelerometer x
# - Column3: accelerometer y
# - Column4: accelerometer z
# - Column5: gyroscope x
# - Column6: gyroscope y
# - Column7: gyroscope z
# - Column8: magnetometer x
# - Column9: magnetometer y
# - Column10: magnetometer z
# - Column11: Timestamp
# - Column12: Activity Label (16 atividades)
# 
#
# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from utils import *


# %%

loc = []

for i in range (0,15):
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


# %%


# act1 = t_acc[array[:,-1]==5]

activities = np.arange(1, 17) # nao sei se util
box_acc = [] # TODO ver se dá para fazer isto com np.array
box_gyr = []
box_mag = []
for i in range(1,17):
    act = (array[:,-1]==i)
    #length = np.sum(act)
    box_acc.append(t_acc[act]) # TODO outra forma?
    box_gyr.append(t_gyr[act])
    box_mag.append(t_mag[act])

#box_acc = np.array(box_acc)
plt.figure()
fig, axs = plt.subplots(3)
axs[0].boxplot(box_acc)
axs[1].boxplot(box_gyr)
axs[2].boxplot(box_mag)
plt.show()

# %%

# desvio e outliers para cada k = 3, 3.5, 4
d = []
outliersk = [] 

x_density = 5 # percentage

# for c, d in zip(a, b) itera alternadamente cada lista no mesmo loop
for box in ([box_acc, box_gyr, box_mag]):
    for i in range(16):
        q1 = np.quantile(box[:][i], 0.25)
        q3 = np.quantile(box[:][i], 0.75)
        #av = np.average(box[:][i])
        
        iqr = q3-q1
        upper_bound = q3+(1.5*iqr)
        lower_bound = q1-(1.5*iqr)
    
        outliers = box[:][i][(box[:][i] <= lower_bound) | (box[:][i] >= upper_bound)]
        out_bool = (box[:][i] <= lower_bound) | (box[:][i] >= upper_bound)
        #print('The following are the outliers in the boxplot:{}'.format(outliers))
        box[:][i] = box[:][i][(box[:][i] >= lower_bound) & (box[:][i] <= upper_bound)]
        
        #unique, counts = np.unique(out_bool, return_counts=True)
        #desvios
        counts = np.count_nonzero(out_bool==True)
        d.append((counts/out_bool.size)*100) # TODO se a coluna nao tiver desvios
        
        zscore = stats.zscore(box[:][i], axis=0, ddof=0, nan_policy='propagate')
        outliersk.append(box[:][i][(zscore <= -3) | (zscore >= 3)])
        #print('The following are the outliers from the z-score test: {}'.format(outliersk[:][i]))
        
        #iterar para cada coluna
        centroids, cluster = k_means(box[:][i], 3)


# %%
t_out, td = get_outliers(t_acc)
p = t_out.shape[0]
acc_out = inject_outliers(10, 4, t_out, p)



# %%
it = 0
for i in range(t_out.size):
    if t_out[i] != acc_out[i]:
        it+=1

print(td)
print(it)

# %%
for vec in box_acc:
    
    n = vec.size    
    ran = np.ptp(vec)
    test = np.random.rand(n)*ran
    #print(np.reshape(test[:n], (-1,1)))
    #print(np.reshape(vec[:n], (-1,1)).shape)
    test = np.append(test[:n], test[:n]).reshape(-1, 2)
    print(vec[:n].reshape(-1,1))
    coef = fit_linear(test, vec, n)
    print(coef)


# %%

for i in range(16):
    stats.kstest(box_acc[:][i], 'norm') # if follows a gaussian


# %%


# %% [markdown]
# 


