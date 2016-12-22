"""
author: Junxian Ye
time: 12/22/2016
link: https://github.com/un-knight/coursera-machine-learning-algorithm
"""

import numpy as np
import pandas as pd
import scipy.io as sio
import sklearn.svm
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
from matplotlib import pyplot as plt


def read_data_from_mat(file=None):
    mat = sio.loadmat(file)
    training = pd.DataFrame(mat['X'], columns=['X1', 'X2'])
    training['y'] = mat['y']

    cv = pd.DataFrame(mat['Xval'], columns=['X1', 'X2'])
    cv['y'] = mat['yval']

    return training, cv


def main():
    training, cv = read_data_from_mat('./data/ex6data3.mat')

    candidate = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
    combination = [(C, sigma) for C in candidate for sigma in candidate]
    search = []

    for C, sigma in combination:
        svc = sklearn.svm.SVC(C=C, gamma=sigma)
        svc.fit(training[['X1', 'X2']], training['y'])
        search.append(svc.score(cv[['X1', 'X2']], cv['y']))

    best_index = np.argmax(search)
    best_score = search[best_index]
    best_param = combination[best_index]
    print(best_score, best_param)

    best_svc = sklearn.svm.SVC(C=best_param[0], gamma=best_param[1])
    best_svc.fit(training[['X1', 'X2']], training['y'])
    y_pred = best_svc.predict(cv[['X1', 'X2']])
    print(metrics.classification_report(y_true=cv['y'], y_pred=y_pred))

    # sklearn GridSearchCV
    parameters = {
        'C': candidate,
        'gamma': candidate
    }
    svc = sklearn.svm.SVC()
    clf = GridSearchCV(svc, parameters, n_jobs=-1)
    clf.fit(training[['X1', 'X2']], training['y'])
    print(clf.best_params_, 'best score: ', clf.best_score_)
    y_pred = clf.predict(cv[['X1', 'X2']])
    print(metrics.classification_report(y_true=cv['y'], y_pred=y_pred))


if __name__ == "__main__":
    main()