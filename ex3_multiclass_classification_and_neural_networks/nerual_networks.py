"""
author: Junxian Ye
time: 12/03/2016
link: https://github.com/un-knight/machine-learning-algorithm
"""
from func.tools import *
from sklearn.metrics import classification_report
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def main():
    theta1, theta2 = read_weights_from_mat('ex3weights.mat')
    print("theta1 size {}".format(theta1.shape))
    print("theta2 size {}".format(theta2.shape))

    X, y = read_data_from_mat('ex3data1.mat', transpose=False)
    print("X size {}".format(X.shape))
    print("y size {}".format(y.shape))
    X_extend = np.column_stack((np.ones((X.shape[0], 1)), X))

    # layer1
    a1 = X_extend

    # layer2
    z2 = a1 @ theta1.T
    print("z2 size {}".format(z2.shape))
    z2 = np.column_stack((np.ones((z2.shape[0], 1)), z2))
    a2 = sigmoid(z2)

    # layer3
    z3 = a2 @ theta2.T
    a3 = sigmoid(z3)

    y_pre = np.argmax(a3, axis=1) + 1
    print(classification_report(y_true=y, y_pred=y_pre))


if __name__ == "__main__":
    main()