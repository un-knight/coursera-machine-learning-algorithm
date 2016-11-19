import numpy as np


def read_from_txt(filename=None):
    """
    read dataset from txt and return np.array
    :param filename: filename
    :return: x, y (type of np.array)
    """
    with open(filename, 'r') as file_to_read:
        x_data = []
        y_data = []
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            # 不足：考虑到数据集通常会采用多个特征所以需要多个 x 与之对应，这里需要探索一种更一般化的解析方法
            x, y = [float(i) for i in lines.split(',')]
            x_data.append(x)
            y_data.append(y)
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        return x_data, y_data
