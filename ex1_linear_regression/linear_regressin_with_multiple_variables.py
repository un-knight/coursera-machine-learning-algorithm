"""
author: Ye Junxian
time: 11/19/2016
link: https://github.com/un-knight/machine-learning-algorithm
"""

from func.read_from_ex1data2 import read_from_txt
import numpy as np
from matplotlib import pyplot as plt


def feature_normalize(data=None):
    if data is None:
        return
    data_norm = np.zeros(data.shape)
    # axis=0 对每一列操作
    # axis=1 对每一行操作
    # 求均值
    mean = np.mean(data, axis=0)
    # 求标准差
    std = np.std(data, axis=0)

    data_norm = (data - mean) / std
    return data_norm, mean, std


def calc_cost(x, y, theta):
    m = x.shape[0]
    h = np.dot(x, theta)
    diff = h - y
    cost = np.dot(diff.T, diff) / (2 * m)
    # 此处有坑！！！
    return cost.flat[0]


def gradient_descent(x, y, iterations, learning_rate):
    m = x.shape[0]
    theta = np.zeros((x.shape[1], 1))
    yield theta, calc_cost(x, y, theta)

    for i in range(iterations):
        # update theta
        tmp_theta = learning_rate / m * (np.dot((np.dot(x, theta) - y).T, x)).T
        theta = theta - tmp_theta
        yield theta, calc_cost(x, y, theta)


def plot_regression_process(cost_history, iterations):
    iter = np.array([i for i in range(100)])
    plt.plot(iter, cost_history[0:100], 'r-', linewidth=2.0)
    plt.xlabel(r"Number of iterations ($\theta=0.1\ \ and\ \ iteration=400$)")
    plt.ylabel('cost J')
    plt.show()


def normal_equation(x, y):
    part1 = np.dot(x.T, x)
    part2 = np.linalg.inv(part1)
    part3 = np.dot(part2, x.T)
    theta = np.dot(part3, y)
    return theta


def main():
    iterations = 400
    learning_rate = 0.1
    test_x = np.array([[1650, 3]])

    data = read_from_txt('ex1data2.txt')
    x = data[:, :-1]
    # 此处有坑！！！
    y = data[:, -1:]

    x_norm, mean, std = feature_normalize(x)
    x_extend = np.column_stack((np.ones((x_norm.shape[0], 1)), x_norm))
    regression_history = list(gradient_descent(x_extend, y, iterations, learning_rate))
    theta = regression_history[-1][0]
    print('Theta computed from gradient descent: \n', theta)
    cost_history = np.array([cost for _, cost in regression_history])
    plot_regression_process(cost_history, iterations)

    # test with gradient descent
    test_x_norm = (test_x - mean) / std
    test_x_norm = np.column_stack((np.ones((test_x.shape[0], 1)), test_x_norm))
    print('predict result with gradient descent: {}'.format(np.dot(test_x_norm, theta)))

    # calculate the theta with normal equation
    x_neq = np.column_stack((np.ones((x.shape[0], 1)), x))
    theta_neq = normal_equation(x_neq, y)

    # using the theta to predict data
    test_x_extend = np.column_stack((np.ones((test_x.shape[0], 1)), test_x))
    print('predict result with normal equation: {}'.format(np.dot(test_x_extend, theta_neq)))

if __name__ == '__main__':
    main()
