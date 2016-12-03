"""
author: Ye Junxian
time: 12/02/2016
link: https://github.com/un-knight/machine-learning-algorithm
"""
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import scipy.io as sio


def read_data_from_mat(file, transpose=True):
    data = sio.loadmat(file)
    X = data.get('X')
    y = data.get('y')

    if transpose:
        X = np.array([im.reshape((20, 20)).T for im in X])
        X = np.array([im.reshape(400) for im in X])

    y = y.reshape(y.shape[0])

    return X, y


def plot_1_img(img):
    # figsize 指定绘制矩形框大小
    fig, ax = plt.subplots(figsize=(2, 2))
    # matplotlib.cm.binary 指定通过灰度图方式显示
    ax[1, 1].matshow(img.reshape((20, 20)), cmap=matplotlib.cm.binary)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.show()


def plot_100_img(X):
    # 假设图像都是方形的
    img_size = int(np.sqrt(X.shape[1]))

    random_index = np.random.choice(np.arange(X.shape[0]), 100)
    random_img = X[random_index, :]

    fig, ax = plt.subplots(nrows=10, ncols=10, sharey=True, sharex=True, figsize=(8, 8))

    for r in range(10):
        for c in range(10):
            ax[r, c].matshow(random_img[10*r + c].reshape((img_size, img_size)),
                             cmap=matplotlib.cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))

    plt.show()


def read_weights_from_mat(file):
    data = sio.loadmat(file)
    theta1 = data.get('Theta1')
    theta2 = data.get('Theta2')
    return theta1, theta2
