import scipy.io as sio
import numpy as np
import scipy.optimize as opt


def read_data_from_mat(file):
    """
    data['X'] has shape of (12, 1), while pandas need (12,).
    So using np.ravel to convert the dataset.
    :param file: filename
    :return: X, y, Xval, yval, Xtest, ytest
    """
    data = sio.loadmat(file)
    return map(np.ravel, [data['X'], data['y'], data['Xval'], data['yval'],
                          data['Xtest'], data['ytest']])

def cost_function(theta, X, y, l=1):
    """
    :param theta:
    :param X:
    :param y:
    :param l: lambda
    :return: cost with regularization
    """
    m = X.shape[0]
    h = X @ theta
    diff = h - y
    cost = 1 / (2*m) * (diff @ diff.T)

    reg = l / (2*m) * (theta[1:] ** 2).sum()
    return cost + reg


def regression_gradient(theta, X, y):
    m = X.shape[0]
    # h = X @ theta
    # gradient = 1. / m * X.T @ (h - y)
    return (X.T @ (X @ theta - y)) / m


def reg_regression_gradient(theta, X, y, l=1.0):
    """
    regularized regression gradient
    :param theta:
    :param X:
    :param y:
    :param l: lambda
    :return:
    """
    m = X.shape[0]
    reg_theta = theta.copy()
    reg_theta[0] = 0
    reg_term = (l / m) * reg_theta
    return regression_gradient(theta, X, y) + reg_term


def linear_regression(X, y, l=1):
    theta = np.ones(X.shape[1])
    options = {'disp': False}
    # options = {'disp': True}

    res = opt.minimize(fun=cost_function, x0=theta, args=(X, y, l),
                       method='TNC', jac=reg_regression_gradient,
                       options=options)
    return res


def normalize_feature(df):
    return df.apply(lambda column: (column - column.mean()) / column.std())