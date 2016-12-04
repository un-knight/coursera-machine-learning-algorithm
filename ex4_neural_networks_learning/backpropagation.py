"""
author: Junxian Ye
time: 12/04/2016
link: https://github.com/un-knight/machine-learning-algorithm
"""
from func.tools import *
import numpy as np
import scipy.optimize as opt
from sklearn.metrics import classification_report
import matplotlib


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))


def expand_y(y):
    k = np.unique(y).shape[0]
    yy = []
    for i in y:
        line = np.zeros(k)
        line[i - 1] = 1
        yy.append(line)
    return np.array(yy)


def expand_array(arr):
    tmp = arr.reshape((1, arr.shape[0]))
    return np.ones(tmp.shape).T @ tmp


def feedforward_propagation(theta, X):
    theta1, theta2 = deserialize(theta)
    m = X.shape[0]

    # input layer
    a1 = X
    # print("a1 size {}".format(a1.shape))

    # hidden layer
    z2 = a1 @ theta1.T
    a2 = sigmoid(z2)
    a2 = np.column_stack((np.ones((a2.shape[0], 1)), a2))
    # print("a2 size {}".format(a2.shape))

    # output layer
    z3 = a2 @ theta2.T
    h = sigmoid(z3)

    return a1, z2, a2, z3, h


def gradient(theta, X, y):
    theta1, theta2 = deserialize(theta)
    m = X.shape[0]
    delta1 = np.zeros(theta1.shape)
    delta2 = np.zeros(theta2.shape)

    a1, z2, a2, z3, h = feedforward_propagation(theta, X)

    for i in range(m):
        a1i = a1[i, :]
        z2i = z2[i, :]
        a2i = a2[i, :]
        hi = h[i, :]
        yi = y[i, :]

        d3i = hi - yi

        z2i = np.insert(z2i, 0, np.ones(1))
        d2i = theta2.T @ d3i * sigmoid_gradient(z2i)

        delta2 += np.matrix(d3i).T @ np.matrix(a2i)
        delta1 += np.matrix(d2i[1:]).T @ np.matrix(a1i)

    delta1 = delta1 / m
    delta2 = delta2 / m

    return serialize(delta1, delta2)


def regularized_gradient(theta, X, y, l=1):
    m = X.shape[0]
    delta1, delta2 = deserialize(gradient(theta, X, y))
    theta1, theta2 = deserialize(theta)

    theta1[:, 0] = 0
    reg_term_d1 = (l / m) * theta1
    delta1 = delta1 + reg_term_d1

    theta2[:, 0] = 0
    reg_term_d2 = (l / m) * theta2
    delta2 = delta2 + reg_term_d2

    return serialize(delta1, delta2)


def cost(theta, X, y):
    m = X.shape[0]
    _, _, _, _, h = feedforward_propagation(theta, X)
    cost = -1 / m * (y*np.log(h) + (1-y)*np.log(1-h)).sum()

    return cost

def regularized_cost(theta, X, y, l=1):
    theta1, theta2 = deserialize(theta)
    m = X.shape[0]

    reg_t1 = l / (2 * m) * (theta1[:, 1:] ** 2).sum()
    reg_t2 = l / (2 * m) * (theta2[:, 1:] ** 2).sum()

    return cost(theta, X, y) + reg_t1 + reg_t2


def gradient_checking(theta, X, y, epsilon, regularized=False):
    def a_numeric_grad(plus, minus, regularized=False):
        if regularized:
            return (regularized_cost(plus, X, y) - regularized_cost(minus, X, y)) / (epsilon * 2)
        else:
            return (cost(plus, X, y) - cost(minus, X, y)) / (epsilon * 2)

    theta_matrix = expand_array(theta)
    epsilon_matrix = np.identity(theta.shape[0]) * epsilon

    plus_matrix = theta_matrix + epsilon_matrix
    minus_matrix = theta_matrix - epsilon_matrix

    numeric_grad = np.array([a_numeric_grad(plus_matrix[i], minus_matrix[i], regularized)
                             for i in range(theta.shape[0])])

    analytic_grad = regularized_gradient(theta, X, y) if regularized else gradient(theta, X, y)

    diff = np.linalg.norm(numeric_grad - analytic_grad) / np.linalg.norm(numeric_grad + analytic_grad)

    print('If your backpropagation implementation is correct,\n'
        'the relative difference will be smaller than 10e-9 (assume epsilon=0.0001).\n'
        'Relative Difference: {}\n'.format(diff))


def random_init(size, epsilon_init=0.12):
    # initializing theta
    return np.random.uniform(-epsilon_init, epsilon_init, size)


def nn_training(X, y):
    theta = random_init(10285)
    options = {
        'maxiter': 400
    }
    res = opt.minimize(fun=regularized_cost, x0=theta, args=(X, y, 1),
                       method='TNC', jac=regularized_gradient, options=options)
    return res

def plot_hidden_layer(theta):
    theta1, _ = deserialize(theta)
    hidden_layer = theta1[:, 1:]

    fig, ax = plt.subplots(nrows=5, ncols=5, sharey=True, sharex=True, figsize=(8, 8))

    for r in range(5):
        for c in range(5):
            ax[r, c].matshow(hidden_layer[5 * r + c].reshape((20, 20)),
                             cmap=matplotlib.cm.binary)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))

    plt.show()

def main():
    X_train, y_train = read_data_from_mat('ex4data1.mat')
    print("X_train size {}\ny_train size {}".format(X_train.shape, y_train.shape))

    X_train_extension = np.column_stack((np.ones((X_train.shape[0], 1)), X_train))
    y_target = expand_y(y_train)
    print("x_train_extension size {}\ny_target size {}".format(X_train_extension.shape,
                                                           y_train.shape))

    # print("sigmoid gradient on 0.5: {}".format(sigmoid_gradient(0)))
    theta1, theta2 = read_weights_from_mat('ex4weights.mat')
    print("theta1 size {}\ntheta2 size {}".format(theta1.shape, theta2.shape))
    theta = serialize(theta1, theta2)
    print("theta size {}".format(theta.shape))
    # delta1, delta2 = deserialize(gradient(theta, X_train_extension, y_target))
    # print("delta1 size {}\ndelta2 size {}".format(delta1.shape, delta2.shape))

    # gradient_checking(theta, X_train_extension, y_target, epsilon=0.0001)
    # gradient_checking(theta, X_train_extension, y_target, epsilon=0.0001, regularized=True)

    res = nn_training(X_train_extension, y_target)
    theta = res.x

    _, _, _, _, h = feedforward_propagation(theta, X_train_extension)
    y_pred = np.argmax(h, axis=1) + 1
    print(classification_report(y_true=y_train, y_pred=y_pred))

    plot_hidden_layer(theta)


if __name__ == '__main__':
    main()