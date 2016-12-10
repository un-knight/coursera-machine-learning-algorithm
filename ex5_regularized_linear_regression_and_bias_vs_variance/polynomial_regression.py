"""
author: Junxian Ye
time: 12/10/2016
link: https://github.com/un-knight/machine-learning-algorithm
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from func import tools as tl


def poly_features(X, power=1, as_ndarray=False):
    data = {'f{}'.format(i): X ** i for i in range(1, power + 1)}
    df = pd.DataFrame(data)
    return df.as_matrix() if as_ndarray else df


def prepare_poly_data2(*args, power=1):
    def prepare(x):
        # expand feature
        df = poly_features(x, power=power)

        # normalisation
        x_array = tl.normalize_feature(df).as_matrix()

        # add intercept term
        return np.column_stack((np.ones((x_array.shape[0], 1)), x_array))

    return [prepare(x) for x in args]


def prepare_poly_data(*args, power=1):
    for x in args:
        # expand feature
        df = poly_features(x, power=power)
        # normalisation
        x_array = tl.normalize_feature(df).as_matrix()
        # add intercept term
        yield np.column_stack((np.ones((x_array.shape[0], 1)), x_array))


def plot_learning_curve(X, y, Xval, yval, l=1):
    error_train, error_val = [], []
    m = X.shape[0]

    for i in range(1, m + 1):
        res = tl.linear_regression(X[:i, :], y[:i], l)
        theta = res.x
        et = tl.cost_function(theta, X[:i, :], y[:i], l)
        ev = tl.cost_function(theta, Xval, yval, l)

        error_train.append(et)
        error_val.append(ev)

    plt.plot(np.arange(1, m + 1), error_train, label='Train')
    plt.plot(np.arange(1, m + 1), error_val, label='Cross Validation')
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.title('Polynomial Reression Learning Curve ($\lambda$ = {})'.format(l))
    plt.legend(loc=1)
    plt.show()


def main():
    X, y, Xval, yval, Xtest, ytest = tl.read_data_from_mat('ex5data1.mat')

    """
    create polynomial features
    """
    # poly_X = poly_features(X, power=1)
    # print(poly_X)

    """
    prepare polynomial regression data
    """
    # X_poly, Xval_poly, Xtest_poly = prepare_poly_data2(X, Xval, Xtest, power=8)
    # print("X_poly[:3, :] = ", X_poly[:3, :])

    X_poly, Xval_poly, Xtest_poly = prepare_poly_data(X, Xval, Xtest, power=8)
    # print("X_poly[:3, :] = ", X_poly[:3, :])

    # overfitting
    plot_learning_curve(X_poly, y, Xval_poly, yval, l=0)

    # set lambda to 1 to regularized the param
    plot_learning_curve(X_poly, y, Xval_poly, yval, l=0.3)

    # set lambda to 100
    plot_learning_curve(X_poly, y, Xval_poly, yval, l=100)

    """
    finding the best lambda
    """
    l_candidate = [0, 0.0001, 0.0003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    error_train, error_val = [], []

    for l in l_candidate:
        res = tl.linear_regression(X_poly, y, l)
        theta = res.x
        et = tl.cost_function(theta, X_poly, y, l)
        ev = tl.cost_function(theta, Xval_poly, yval, l)

        error_train.append(et)
        error_val.append(ev)

    plt.plot(l_candidate, error_train, label='Train')
    plt.plot(l_candidate, error_val, label='Cross Validation')
    plt.xlabel('$\lambda$')
    plt.ylabel('Error')
    plt.legend(loc=2)
    plt.show()


if __name__ == '__main__':
    main()