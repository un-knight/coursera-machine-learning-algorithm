"""
author: Ye Junxian
time: 11/14/2016
link: https://github.com/un-knight/machine-learning-algorithm
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from func.read_from_txt import read_from_txt
from mpl_toolkits.mplot3d import Axes3D


def plot_dataset_scatter(y, x):
    plt.scatter(x, y, marker='x')
    plt.xlim(4, 24)
    plt.xticks(np.linspace(4, 24, 10, endpoint=False))

    plt.xlabel('Population of city in 10,000s')
    plt.ylabel('Profit in 10,000s')

    plt.show()


def plot_result(y, x, theta_history=None, gif_flag=False):
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    fig.set_size_inches((8, 6))
    save_dpi = 80

    plt.scatter(x, y, marker='x')
    plt.xlim(4, 24)
    plt.xticks(np.linspace(4, 24, 10, endpoint=False))
    ax.set_xlabel('Population of city in 10,000s')
    ax.set_ylabel('Profit in 10,000s')
    if theta_history and gif_flag:
        theta_1, theta_0 = theta_history[0]
        line,  = ax.plot(x, theta_0 + theta_1 * x, 'r-', linewidth=2.0)

        def update(frame_i):
            theta_1i, theta_0i = theta_history[frame_i * 2]
            line.set_ydata(theta_0i + theta_1i * x)
            ax.set_title('Fit at iteration {0}'.format(frame_i * 2))
            return [line]

        anim = FuncAnimation(fig, update, frames=range(len(theta_history) // 2), interval=200)
        anim.save('regression_process.gif', dpi=save_dpi, writer='imagemagick')
    else:
        ax.set_title('Fit at iteration {0}'.format(len(theta_history)-1))
        ax.plot(x, theta_history[-1][1] + theta_history[-1][0] * x, 'r-', linewidth=2.0)
        fig.savefig('result.png', dpi=save_dpi)
    plt.show()


def plot_cost_3d(y, x, costfunc, theta_history=None):
    N = 500
    theta_1s = np.linspace(-1.0, 4.0, N)
    theta_0s = np.linspace(-10.0, 10.0, N)
    cost = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            cost[i, j] = costfunc(y, x,
                                  theta_1s[i],
                                  theta_0s[j])

    # 绘制3D 图像
    fig = plt.figure()
    fig.set_tight_layout(True)
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.set_xlabel('theta_1')
    ax1.set_ylabel('theta_0')
    ax1.set_title('Surface')
    theta_1s_grid, theta_0s_grid = np.meshgrid(theta_1s, theta_0s)
    surf = ax1.plot_surface(theta_1s_grid, theta_0s_grid, cost, cmap=cm.coolwarm)


    # 绘制等高线图
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.contour(theta_1s_grid, theta_0s_grid, cost)
    ax2.set_xlabel('theta_1')
    ax2.set_ylabel('theta_0')

    plt.plot(theta_history[-1][0], theta_history[-1][1], 'rx')
    plt.show()


def calc_cost(y, x, theta_1, theta_0):
    """
    y = theta_0 + theta_1 * x
    """
    h = theta_1 * x + theta_0
    d = h - y
    cost = np.dot(d.T, d) / (2*x.shape[0])
    return cost.flat[0]


def gradient_descent(y, x, iterations, learning_rate=0.01):
    m = x.shape[0]
    theta_1, theta_0 = 0, 0
    yield theta_1, theta_0, calc_cost(y, x, theta_1, theta_0)

    # 迭代训练
    for i in range(iterations):
        h = theta_0 + theta_1 * x
        d = h - y
        theta_0 -= learning_rate * d.sum() / m
        theta_1 -= learning_rate * (d * x).sum() / m
        yield theta_1, theta_0, calc_cost(y, x, theta_1, theta_0)


def main():
    # 常数
    learning_rate = 0.01
    iterations = 1500  # 迭代次数
    n = 500  # 生成样本数
    anim = True  # 是否使用动画展示拟合过程

    x, y = read_from_txt('ex1data1.txt')
    # 绘制数据散点图
    # plot_dataset_scatter(y, x)

    history = list(gradient_descent(y, x, iterations, learning_rate))
    theta_history = [(theta_1, theta_0) for theta_1, theta_0, _ in history]
    print('theta_1: {0}, theta_0: {1}, cost: {2}'.format(history[-1][0], history[-1][1], history[-1][2]))

    plot_result(y, x, theta_history, anim)
    plot_cost_3d(y, x, calc_cost, theta_history)


if __name__ == '__main__':
    main()
