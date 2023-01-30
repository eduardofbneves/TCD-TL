from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from math import pi, cos, sin
from scipy.fft import fft, fftfreq, fftshift

""" dataset = [
    [106, 210, 120, 80, 70, 60, 220, 155, 230],
    [100, 140, 180, 160, 55, 45, 78, 120, 155],
    [267, 371, 305, 225, 123, 100, 302, 279, 389],
    [75, 89, 80, 45, 123, 89, 99, 100, 67]
]
dataset = pd.DataFrame(np.array(dataset), index={'BP', 'HR', 'LVET', 'PEP'})
labels = ['N', 'N', 'N', 'HF', 'HF', 'HF', 'N', 'N', 'HF']
target_colors = ['b', 'b', 'b', 'r', 'r', 'r', 'b', 'b', 'r']

# normalizar/centrar os dados
dataset = np.round(((dataset - dataset.mean()) / dataset.std()).T, decimals=2)

pca = PCA(n_components=4)
output = pd.DataFrame(
    pca.fit_transform(dataset),
    columns={'PC1', 'PC2', 'PC3', 'PC4'}
)
pervar = np.round(pca.explained_variance_ratio_ * 100, decimals=2)
print('eigenvectors >> \n', np.round(pca.components_, decimals=2))
print('eigenvalues >> \n', np.round(pca.singular_values_ ** 2, decimals=2))
print('evr% >> ', pervar)

plt.figure(figsize=(5, 5))
plt.scatter(output.iloc[:, 0], output.iloc[:, 1],
            c=target_colors, cmap='plasma')
plt.xlabel('pc1')
plt.ylabel('pc2')
plt.show()


# import relevant libraries for 3d graph
fig = plt.figure(figsize=(5, 5))

# choose projection 3d for creating a 3d graph
axis = fig.add_subplot(111, projection='3d')
 """
# x[:,0]is pc1,x[:,1] is pc2 while x[:,2] is pc3
""" axis.scatter(output.iloc[:, 0], output.iloc[:, 1],
             output.iloc[:, 2], c=target_colors, cmap='plasma')
axis.set_xlabel("PC1", fontsize=10)
axis.set_ylabel("PC2", fontsize=10)
axis.set_zlabel("PC3", fontsize=10)
plt.show() """

""" pca = PCA(0.75)
output = pca.fit_transform(dataset)
# output = pd.DataFrame(output, columns={'PC1', 'PC2', 'PC3', 'PC4'})
pervar = np.round(pca.explained_variance_ratio_ * 100, decimals=2)
print('eigenvectors >> \n', np.round(pca.components_, decimals=2))
print('eigenvalues >> \n', np.round(pca.singular_values_ ** 2, decimals=2))
print('evr% >> ', pervar) """

Y = [
    1.0872, 0.1347, 0.0847,
    0.6020, 0.7925, 0.1781,
    -0.7778, -1.1272, -0.4220,
    0.7722
]

t = np.arange(0, 90, 10)
sp = fftshift(fft(Y))
freq = fftshift(fftfreq(t.shape[-1]))
print("sp >>", sp)
print("freq >>", freq)
