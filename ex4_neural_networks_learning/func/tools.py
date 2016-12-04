"""
author: Junxian Ye
time:
link:
"""
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import scipy.io as sio


def read_data_from_mat(file, transpose=True):
    data = sio.loadmat(file)
    X = data.get("X")
    y = data.get("y")

    if transpose:
        X = np.array([k.reshape(20, 20).T for k in X])
        X = np.array([k.reshape(400) for k in X])

    y = y.reshape(y.shape[0])

    return X, y


def read_weights_from_mat(file):
    data = sio.loadmat(file)
    theta1 = data.get("Theta1")
    theta2 = data.get("Theta2")

    return theta1, theta2


def plot_100_images(X):
    # 假设图像都是方形的
    img_size = int(np.sqrt(X.shape[1]))

    random_index = np.random.randint(0, 5000, 100)
    random_img = X[random_index, :]

    fig, ax = plt.subplots(nrows=10, ncols=10, sharex=True, sharey=True,
                           figsize=(8, 8))

    for r in range(10):
        for c in range(10):
            ax[r, c].matshow(random_img[10*r + c].reshape((img_size, img_size)),
                             cmap=matplotlib.cm.binary)
            # 消去坐标刻度
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))

    plt.show()