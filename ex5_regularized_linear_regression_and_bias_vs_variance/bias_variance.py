"""
author: Junxian Ye
time: 12/10/2016
link: https://github.com/un-knight/machine-learning-algorithm
"""
import numpy as np
from matplotlib import pyplot as plt

from func import tools as tl


def main():
    X, y, Xval, yval, Xtest, ytest = tl.read_data_from_mat('ex5data1.mat')
    # changing the shape and add bias term for each data
    X, Xval, Xtest = [x.reshape(x.shape[0], 1) \
                      for x in (X, Xval, Xtest)]
    X_extension, Xval_extension, Xtest_extension = [np.column_stack((np.ones((x.shape[0], 1)), x)) \
                                                    for x in (X, Xval, Xtest)]

    """
    learning curves
    """
    error_train, error_val = [], []
    m = X_extension.shape[0]
    for i in range(1, m+1):
        # setting params l=0 to hidden regularization term
        res = tl.linear_regression(X_extension[:i, :], y[:i], l=0)
        theta = res.x
        et = tl.cost_function(theta, X_extension[:i, :], y[:i], l=0)
        ev = tl.cost_function(theta, Xval_extension, yval, l=0)

        error_train.append(et)
        error_val.append(ev)

    plt.plot(np.arange(1, m + 1), error_val, label='Cross Validation')
    plt.plot(np.arange(1, m+1), error_train, label='Train')
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.legend(loc=1)
    plt.show()


if __name__ == '__main__':
    main()
