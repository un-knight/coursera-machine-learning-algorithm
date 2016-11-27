"""
author: Ye Junxian
time: 11/20/2016
link: https://github.com/un-knight/machine-learning-algorithm
"""

from func.read_from_txt import read_from_txt
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize


def feature_normalize(data=None):
    if data is None:
        return
    else:
        mu = np.mean(data, axis=0)
        sigma = np.std(data, axis=0)
        data_norm = (data - mu) / sigma
        return data_norm, mu.flat[0], sigma.flat[0]


def plot_boundary(x, y, x1_points, x2_points):
    # 这里后期需要考虑进行代码优化，精简代码
    pos_x = []
    pos_y = []
    neg_x = []
    neg_y = []
    for i, yi in enumerate(y):
        if yi == 1:
            pos_x.append(x[i, 0])
            pos_y.append(x[i, 1])
        else:
            neg_x.append(x[i, 0])
            neg_y.append(x[i, 1])
    pos_x = np.array(pos_x)
    pos_y = np.array(pos_y)
    neg_x = np.array(neg_x)
    neg_y = np.array(neg_y)
    plt.scatter(pos_x, pos_y, marker='+', color='red', s=50.0, label='Admitted')
    plt.scatter(neg_x, neg_y, marker='.', color='blue', s=50.0, label='Not admitted')

    plt.xlim(25, 105)
    plt.ylim(25, 105)
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend(loc='upper right')
    plt.title('Boundary')

    plt.plot(x1_points, x2_points, linestyle='-', linewidth=2.0, color='red')

    plt.show()


def sigmoid_function(x, theta):
    z = np.dot(x, theta)
    g = 1.0 / (1.0 + np.exp(-z))
    return g


def calc_cost(x, y, theta):
    m = x.shape[0]
    h = sigmoid_function(x, theta)
    pos_part = np.dot(y.T, np.log(h))
    neg_part = np.dot((1 - y).T, np.log((1 - h)))
    cost = -1.0 / m * (pos_part + neg_part)

    return cost.flat[0]


def gradient_descent(x, y, iterations=400, learning_rate=0.1):
    m = x.shape[0]
    theta = np.zeros((x.shape[1], 1))
    yield theta, calc_cost(x, y, theta)

    for i in range(iterations):
        h = sigmoid_function(x, theta)
        diff = h - y
        derivative = 1 / m * learning_rate * np.dot(diff.T, x).T
        theta -= derivative
        yield theta, calc_cost(x, y, theta)


def main():
    iterations = 400
    learning_rate = 3.0

    data_train = read_from_txt('ex2data1.txt')
    x_train = data_train[:, :-1]
    y_train = data_train[:, -1:]

    x_train_normalized, mu, sigma = feature_normalize(x_train)
    x_train_extend = np.column_stack((np.ones((x_train.shape[0], 1)), x_train_normalized))

    history = list(gradient_descent(x_train_extend, y_train, iterations, learning_rate))
    theta = history[-1][0]
    cost_history = [cost for _, cost in history]
    print('iterations: {0}, learning_rate: {1}'.format(iterations, learning_rate))
    print('theta: ', theta)
    # print('cost: ', cost_history)

    # predict with exam1 = 45 and exam2 = 85
    x_test = np.array([[45.0, 85.0]])
    x_test_norm = (x_test - mu) / sigma
    x_test_extend = np.column_stack((np.ones((1, 1)), x_test_norm))
    z = np.dot(x_test_extend, theta)
    print('prediect exam1=45 and exam2=85: ', 1.0 / (1.0 + np.exp(-z)))

    # plot boundary
    index_min_x1 = np.argmin(x_train[:, 0])
    index_max_x1 = np.argmax(x_train[:, 0])
    boundary_x1_point = [x_train[index_min_x1, 0] - 10, x_train[index_max_x1, 0] + 10]
    boundary_x1_point_norm = [(x - mu) / sigma for x in boundary_x1_point]
    boundary_x2_point_norm = [-(theta[0] + theta[1] * x) / theta[2] for x in boundary_x1_point_norm]
    boundary_x2_point = [x * sigma + mu for x in boundary_x2_point_norm]
    boundary_x1_point = np.array(boundary_x1_point)
    boundary_x2_point = np.array(boundary_x2_point)
    plot_boundary(x_train, y_train, boundary_x1_point, boundary_x2_point)

if __name__ == '__main__':
    main()
