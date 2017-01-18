"""
author: Junxian Ye
time: 01/17/2017
link: https://github.com/un-knight/coursera-machine-learning-algorithm
"""

import seaborn as sns
import numpy as np
import scipy.io as sio
import pandas as pd
from matplotlib import pyplot as plt


def normalize(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

def covariance_matrix(X):
    m = X.shape[0]
    return X.T @ X / m

def project_data(X, U, k):
    m, n = X.shape
    if k > n:
        raise ValueError("k must less than n")
    return X @ U[:, :k]

def reconstruct(Z, U):
    m, n = Z.shape
    return Z @ U[:, :n].T

def main():
    X = sio.loadmat("./data/ex7data1.mat").get('X')
    sns.lmplot('X1', 'X2',
               data=pd.DataFrame(data=X, columns=['X1', 'X2']),
               fit_reg=False)
    # normalize data
    X_norm = normalize(X)
    sns.lmplot('X1', 'X2',
               data=pd.DataFrame(data=X_norm, columns=['X1', 'X2']),
               fit_reg=False)
    plt.show()

    Sigma = covariance_matrix(X_norm)
    # print('Sigma: ', Sigma)
    U, _, _ = np.linalg.svd(Sigma)
    # u1 = U[0]
    # print('u1: ', u1)

    # project data
    Z = project_data(X_norm, U, k=1)
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 4))
    sns.regplot('X1', 'X2',
               data=pd.DataFrame(data=X_norm, columns=['X1', 'X2']),
               fit_reg=False, ax=ax1)
    ax1.set_title('Original')

    sns.rugplot(Z, ax=ax2)
    ax2.set_title('Z dimension')
    ax2.set_xlabel('Z')
    plt.show()

    X_approx = reconstruct(Z, U)

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12, 4))
    sns.rugplot(Z, ax=ax1)
    ax1.set_title("Z dimension")
    ax1.set_xlabel("Z")

    sns.regplot('X1', 'X2',
                data=pd.DataFrame(X_approx, columns=['X1', 'X2']),
                fit_reg=False,
                ax=ax2)
    ax2.set_title("Reconstruct from Z")

    sns.regplot('X1', 'X2',
                data=pd.DataFrame(X_norm, columns=['X1', 'X2']),
                fit_reg=False,
                ax=ax3)
    ax3.set_title("Original dimension")
    plt.show()

if __name__ == '__main__':
    main()