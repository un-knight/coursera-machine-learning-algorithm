"""
author: Junxian Ye
time: 12/04/2016
link: https://github.com/un-knight/machine-learning-algorithm
"""
from func.tools import *
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def expand_y(y, k):
    """
    为方便计算，将 y 扩展为 (5000, 10)
    :param y: 数据 y
    :param k: 种类 k
    :return: 扩展后的数据 y
    """
    yy = []
    for i in y:
        line = np.zeros(k)
        line[i - 1] = 1
        yy.append(line)

    return np.array(yy)


def feedforward_propagation(theta1, theta2, X):
    # input layer
    X_train_extend = np.column_stack((np.ones((X.shape[0], 1)), X))
    a1 = X_train_extend
    print("a1 size {}".format(a1.shape))

    # hidden layer
    z2 = a1 @ theta1.T
    a2 = sigmoid(z2)
    print("a2 size {}".format(a2.shape))
    a2 = np.column_stack((np.ones((a2.shape[0], 1)), a2))

    # output layer
    z3 = a2 @ theta2.T
    a3 = sigmoid(z3)
    print("a3 size {}".format(a3.shape))

    return a3


def cost(h, y):
    m = h.shape[0]
    return -1.0 / m * (y*np.log(h) + (1-y)*np.log(1-h)).sum()


def cost_with_regularization(theta1, theta2, h, y, l=1):
    """
    :param l: lamda
    """
    m = h.shape[0]
    theta1_reg = theta1[:, 1:]
    theta2_reg = theta2[:, 1:]
    regularized_term = (l/(2*m)) * ((theta1_reg ** 2).sum() + (theta2_reg ** 2).sum())

    return cost(h, y) + regularized_term


def main():
    # X, y = read_data_from_mat('ex4data1.mat')
    # print("X size {}\ny size{}".format(X.shape, y.shape))

    # plot_100_images(X)

    theta1, theta2 = read_weights_from_mat('ex4weights.mat')
    print("theta1 size {}\ntheta2 size{}".format(theta1.shape, theta2.shape))
    # print("theta1", theta1)
    # print("theta2", theta2)

    X_train, y_train = read_data_from_mat('ex4data1.mat', transpose=False)
    print("X_train size {}\ny_train size{}".format(X_train.shape, y_train.shape))

    k_array = np.unique(y_train)
    k = k_array.shape[0]
    print("{} classes".format(k))

    y_target = expand_y(y_train, k)
    print("y_target size {}".format(y_target.shape))

    # feedforward propagation
    h = feedforward_propagation(theta1, theta2, X_train)
    c = cost(h, y_target)
    print("cost: {}".format(c))
    cr = cost_with_regularization(theta1, theta2, h, y_target)
    print("cost with regularization: {}".format(cr))


if __name__ == '__main__':
    main()