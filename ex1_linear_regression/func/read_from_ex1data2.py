import numpy as np


def read_from_txt(filename=None):
    """
    read dataset from txt and return np.array
    :param filename: filename
    :return: x, y (type of np.array)
    """
    with open(filename, 'r') as file_to_read:
        data = []
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            data_line = [float(i) for i in lines.split(',')]
            data.append(data_line)
        data = np.array(data)
        return data
