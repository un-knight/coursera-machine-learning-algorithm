from matplotlib import pyplot as plt
import numpy as np


def plot_data(x, y):
    """
    plot scatter
    :param x: x data
    :param y: y data
    :return: None
    """
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
    plt.scatter(pos_x, pos_y, marker='+', color='red', s=50.0, label='y=1')
    plt.scatter(neg_x, neg_y, marker='.', color='blue', s=50.0, label='y=0')

    plt.xlim(-1, 1.5)
    plt.ylim(-0.8, 1.2)
    plt.xlabel('Test 1')
    plt.ylabel('Test 2')
    plt.legend(loc='upper right')
    plt.title('Scatter')

    plt.show()
