"""
author: Junxian Ye
time: 12/10/2016
link: https://github.com/un-knight/machine-learning-algorithm
"""

import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

from func import tools as tl


def main():
    X, y, Xval, yval, Xtest, ytest = tl.read_data_from_mat('ex5data1.mat')

    """
    Visualizing the data

    Here using the seaborn to plot the dataset.
    """
    df = pd.DataFrame({'Change in water level (x)': X,
                       'Water flowing out of the dam (y)': y})
    # print(df)
    sns.lmplot('Change in water level (x)', 'Water flowing out of the dam (y)',
               data=df, fit_reg=False, size=7)
    plt.show()

    """
    Regularized linear regression cost function
    """
    # changing the shape and add bias term for each data
    X, Xval, Xtest = [x.reshape(x.shape[0], 1) \
                      for x in (X, Xval, Xtest)]
    X_extension, Xval_extension, Xtest_extension = [np.column_stack((np.ones((x.shape[0], 1)), x)) \
                                                    for x in (X, Xval, Xtest)]
    # initialized theta with one
    theta = np.ones(X_extension.shape[1])
    print("regularized cost when thetas are one: ", tl.cost_function(theta, X_extension, y))

    """
    caculating regularized gradient
    """
    gra = tl.regression_gradient(theta, X_extension, y)
    print("gradient when thetas are one: ", gra)

    """
    fitting linear regression
    """
    final_theta = tl.linear_regression(X_extension, y, l=0).get('x')

    plt.scatter(X, y, label="Training data")
    plt.plot(X, X_extension @ final_theta, label="Prediction")
    plt.legend(loc=2)
    plt.show()


if __name__ == '__main__':
    main()
