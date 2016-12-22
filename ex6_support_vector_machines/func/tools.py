import numpy as np
import pandas as pd
import scipy.io as sio


def read_data_from_mat(file=None):
    mat = sio.loadmat(file)
    # print(mat.keys())
    # print('X= {}\ny= {}'.format(mat['X'], mat['y']))
    data = pd.DataFrame(mat['X'], columns=['X1', 'X2'])
    data['y'] = mat['y']
    return data