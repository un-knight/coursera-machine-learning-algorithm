"""
author: Junxian Ye
time: 01/03/2017
link:
"""
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats
from sklearn.cross_validation import train_test_split
from matplotlib import pyplot as plt

import anomaly


def main():
    # Loading mat
    mat_data = sio.loadmat('./data/ex8data1.mat')
    # print('ex8data1 key', mat_data.keys())
    X = mat_data.get('X')
    X_val, X_test, y_val, y_test = train_test_split(mat_data.get('Xval'),
                                                    mat_data.get('yval').ravel(),
                                                    test_size=0.5)
    data = pd.DataFrame(X, columns=['Latency', 'Throughput'])
    # sns.regplot('Latency', 'Throughput', data=data, fit_reg=False,
    #            scatter_kws={'s': 30, 'alpha': 0.5})
    # plt.show()

    mu = X.mean(axis=0)
    cov = np.cov(X.T)

    # create multi-var Gaussian model
    multi_normal = stats.multivariate_normal(mu, cov)
    # create a grid
    x, y = np.mgrid[:30:0.1, :30:0.1]
    pos = np.dstack((x, y))

    fig, ax = plt.subplots()
    ax.contourf(x, y, multi_normal.pdf(pos), cmap='Reds')
    sns.regplot('Latency', 'Throughput',
                data=data,
                fit_reg=False,
                ax=ax,
                scatter_kws={"s": 10,
                             "alpha": 0.4})
    plt.show()

    e, fs = anomaly.select_threshold(X, X_val, y_val)
    print('Best epsilon: {}\nBest F-score on validation data: {}'.format(e, fs))

    multi_normal, y_pred = anomaly.predict(X, X_val, e, X_test, y_test)
    # construct test DataFrame
    data = pd.DataFrame(X_test, columns=['Latency', 'Throughput'])
    data['y_pred'] = y_pred

    # create a grid for graphing
    x, y = np.mgrid[0:30:0.01, 0:30:0.01]
    pos = np.dstack((x, y))

    fig, ax = plt.subplots()

    # plot probability density
    ax.contourf(x, y, multi_normal.pdf(pos), cmap='Blues')

    # plot original Xval points
    sns.regplot('Latency', 'Throughput',
                data=data,
                fit_reg=False,
                ax=ax,
                scatter_kws={"s": 10,
                             "alpha": 0.4})

    # mark the predicted anamoly of CV data. We should have a test set for this...
    anamoly_data = data[data['y_pred'] == 1]
    ax.scatter(anamoly_data['Latency'], anamoly_data['Throughput'], marker='x', s=50)
    plt.show()

    '''-----------------------------------------------------------------------------------------'''
    # Part2: high dimension data
    mat_data2 = sio.loadmat('./data/ex8data2.mat')
    X = mat_data2.get('X')
    Xval, Xtest, yval, ytest = train_test_split(mat_data2.get('Xval'),
                                                mat_data2.get('yval').ravel(),
                                                test_size=0.5)
    e, fs = anomaly.select_threshold(X, Xval, yval)
    print('Best epsilon: {}\nBest F-score on validation data: {}'.format(e, fs))
    print('find {} anamolies'.format(y_pred.sum()))

if __name__ == '__main__':
    main()