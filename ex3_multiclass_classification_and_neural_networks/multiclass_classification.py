"""
author: Junxian Ye
time: 12/03/2016
link: https://github.com/un-knight/machine-learning-algorithm
"""
from func.tools import *
import numpy as np
import scipy.optimize as opt

from sklearn.metrics import classification_report


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def calc_cost(theta, x, y):
    h = sigmoid(x @ theta)
    m = x.shape[0]
    cost = -np.mean(y*np.log(h) + (1-y)*np.log(1-h))
    # cost = -1 / m * (y*np.log(h) + (1-y)*np.log(1-h))
    return cost

def predict(x, theta):
    p = sigmoid(x @ theta)
    return (p >= 0.5).astype(int)

def regularized_cost(theta, x, y, l=1):
    theta_j1_to_n = theta[1:]
    m = x.shape[0]
    regularized_term = (l/(2*m)) * (theta_j1_to_n ** 2).sum()

    return calc_cost(theta, x, y) + regularized_term

def gradient(theta, x, y):
    m = x.shape[0]
    h = sigmoid(x @ theta)
    return (1 / m) * x.T @ (h - y)

def regularized_gradient(theta, x, y, l=1):
    theta_j1_to_n = theta[1:]
    m = x.shape[0]
    regularized_theta = l / m * theta_j1_to_n
    regularized_term = np.concatenate([np.array([0]), regularized_theta])
    # print("regularized_term size {}".format(regularized_term.shape))

    return gradient(theta, x, y) + regularized_term

def logistic_regression(x, y, l=1):
    theta = np.zeros(x.shape[1])

    res = opt.minimize(fun=regularized_cost, x0=theta, args=(x, y, l),
                       method='TNC', jac=regularized_gradient, options={'disp': True})

    theta = res.x

    return theta


def main():
    x_train, y_train = read_data_from_mat('ex3data1.mat')

    print("X size {}".format(x_train.shape))
    print("y size {}".format(y_train.shape))

    """
    index = np.random.randint(0, 5000)
    plot_1_img(x_train[index, :])
    print("the number is {}".format(y_train[index]))
    """
    # visualizing the data
    # plot_100_img(x_train)

    # one vs. all classification
    x_train_extend = np.column_stack((np.ones((x_train.shape[0], 1)), x_train))
    print("x_train_extend size {}".format(x_train_extend.shape))

    # 由于在 matlab 中索引从 1 开始，因此所给数据使用 10 来表示数字 0
    # 为了方便使用，这里将类别为 10 的数据挪到索引为 0 的行
    yy = []
    for i in range(1, 11):
        yy.append((y_train == i).astype(int))

    yy = [yy[-1]] + yy[:-1]
    yy = np.array(yy)
    print("yy size {}".format(yy.shape))

    """
    theta_for_0 = logistic_regression(x_train_extend, yy[0])
    print(theta_for_0.shape)
    y_pre = predict(x_train_extend, theta_for_0)
    print("Accuracy: {}".format(np.mean(yy[0] == y_pre)))
    """

    # multiclass classification
    k_theta = np.array([logistic_regression(x_train_extend, yy[k]) for k in range(10)])
    print("k_theta size {}".format(k_theta.shape))

    p = sigmoid(x_train_extend @ k_theta.T)
    np.set_printoptions(suppress=True)
    y_pre = np.argmax(p, axis=1)
    y_target = y_train.copy()
    y_target[y_target == 10] = 0
    print(classification_report(y_target, y_pre))


if __name__ == '__main__':
    main()