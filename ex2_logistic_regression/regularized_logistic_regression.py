"""
author: Ye Junxian
time: 11/23/2016
link: https://github.com/un-knight/machine-learning-algorithm
"""

from func.read_from_txt import read_from_txt
from func.plot_tools import plot_data
import numpy as np
from matplotlib import pyplot as plt





def main():
    data = read_from_txt('ex2data2.txt')
    x_train = data[:, :-1]
    y_train = data[:, -1:]
    plot_data(x_train, y_train)




if __name__ == '__main__':
    main()