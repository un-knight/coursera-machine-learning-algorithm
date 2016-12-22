"""
author: Junxian Ye
time: 12/22/2016
link: https://github.com/un-knight/coursera-machine-learning-algorithm
"""

import numpy as np
import pandas as pd
import sklearn.svm
import seaborn as sns
from matplotlib import pyplot as plt

from func import tools


def gaussian_kernel(x1, x2, sigma=1.0):
    diff = x1 - x2
    k = np.exp(-1 * (diff ** 2).sum() / (2 * (sigma ** 2)))
    return k

def main():
    """
    # test gaussian kernel
    x1 = np.array([1, 2, 1])
    x2 = np.array([0, 4, -1])
    print(gaussian_kernel(x1, x2, 2.0))
    """

    data = tools.read_data_from_mat('./data/ex6data2.mat')

    # Visulize ex6data2.mat
    sns.set(style="white", palette=sns.diverging_palette(240, 10, n=2))
    sns.lmplot('X1', 'X2', hue='y', data=data, size=10, fit_reg=False,
               scatter_kws={'s': 30})
    plt.title('Example Datast 2')
    # plt.show()

    svc = sklearn.svm.SVC(C=100.0, kernel='rbf', gamma=10, probability=True)
    svc.fit(data[['X1', 'X2']], data['y'])
    print('svc score: ', svc.score(data[['X1', 'X2']], data['y']))

    predict_prob = svc.predict_proba(data[['X1', 'X2']])[:, 0]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(data['X1'], data['X2'], s=30, c=predict_prob, cmap='Reds')
    ax.set_title('SVM(Gaussian Kernel) Decision Boundary(Example Dataset 2)')
    plt.show()


if __name__ == '__main__':
    main()