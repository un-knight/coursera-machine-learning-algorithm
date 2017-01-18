"""
author: Junxian Ye
time: 01/17/2017
link: https://github.com/un-knight/coursera-machine-learning-algorithm
"""

import numpy as np
import scipy.io as sio

from sklearn.decomposition import PCA
from matplotlib import pyplot as plt


def plot_image(X, n, title=None):
    img_size = int(np.sqrt(X.shape[1]))
    grid_size = int(np.sqrt(n))

    display = X[:n, :]
    fig, ax = plt.subplots(nrows=grid_size, ncols=grid_size,
                           sharex=True, sharey=True, figsize=(8, 8))
    for i in range(grid_size):
        for j in range(grid_size):
            ax[i, j].imshow(display[i*grid_size+j].reshape(img_size, img_size), )
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))

def main():
    X = sio.loadmat("./data/ex7faces.mat").get('X')
    X = np.array([x.reshape(32, 32).T.reshape(1024) for x in X])
    # print(X[0, :].reshape(32,32))
    plot_image(X, n=64)

    # use pca on sklearn
    pca = PCA(n_components=100)
    Z = pca.fit_transform(X)
    print(Z.shape)
    plot_image(Z, n=64)

    X_approx = pca.inverse_transform(Z)
    plot_image(X_approx, n=64)
    plt.show()

if __name__ == '__main__':
    main()