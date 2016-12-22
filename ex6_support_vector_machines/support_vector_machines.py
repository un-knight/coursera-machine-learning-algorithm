"""
author: Junxian Ye
time: 12/22/2016
link: https://github.com/un-knight/coursera-machine-learning-algorithm
"""

import numpy as np
import pandas as pd
import sklearn.svm
import seaborn as sns
import scipy.io as sio
from matplotlib import pyplot as plt

from func import tools


def main():
    # Visualize ex6data1.mat
    data = tools.read_data_from_mat('./data/ex6data1.mat')

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(data['X1'], data['X2'], s=50, c=data['y'], cmap='Reds')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_title('Example Dataset 1')
    # plt.show()

    # learn by sklearn.svm.LinearSVC with C=1.0
    svc1 = sklearn.svm.LinearSVC(C=1.0, loss='hinge')
    svc1.fit(data[['X1', 'X2']], data['y'])
    print('svc1 score: ', svc1.score(data[['X1', 'X2']], data['y']))

    # Distance of the samples X to the separating hyperplane.
    data['SVM1 confidence'] = svc1.decision_function(data[['X1', 'X2']])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(data['X1'], data['X2'], s=50, c=data['SVM1 confidence'], cmap='RdBu')
    ax.set_title('SVM Descision Boundary with C = 1.0')
    # plt.show()

    # learn by sklearn.svm.LinearSVC with C=100.0
    svc100 = sklearn.svm.LinearSVC(C=100.0, loss='hinge')
    svc100.fit(data[['X1', 'X2']], data['y'])
    print('svc100 score: ', svc100.score(data[['X1', 'X2']], data['y']))

    data['SVM100 confidence'] = svc100.decision_function(data[['X1', 'X2']])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(data['X1'], data['X2'], s=50, c=data['SVM100 confidence'], cmap='RdBu')
    ax.set_title('SVM Decision Boundary with C = 100.0')
    plt.show()


if __name__ == "__main__":
    main()