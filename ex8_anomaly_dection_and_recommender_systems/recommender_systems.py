"""
author: Junxian Ye
time: 05/03/2017
link:
"""

import numpy as np
import seaborn as sns
import pandas as pd
import scipy.io as sio
import scipy.optimize as opt

from matplotlib import pyplot as plt
import recommender as rc


mat_data_path = './data/ex8_movies.mat'
movie_params_path = './data/ex8_movieParams.mat'

def main():
    mat_data = sio.loadmat(mat_data_path)
    # print(mat_data.keys())
    # ex8_movies.mat has two keys 'R' and 'Y'

    Y, R = mat_data.get('Y'), mat_data.get('R')
    # print(Y[0, :], R[0, :])
    num_movies, num_users = Y.shape

    param_mat = sio.loadmat(movie_params_path)
    theta, X = param_mat.get('Theta'), param_mat.get('X')
    num_features = X.shape[1]

    param = rc.serialize(X, theta)
    # print(rc.cost(param, Y, R, num_features))

    # Gradient
    X_grad, theta_grad = rc.deserialize(rc.gradient(param, Y, R, num_features),
                                        num_movies, num_users, num_features)
    # Regularized cost
    # print(rc.regularized_cost(param, Y, R, num_features, l=1))

    # Regularized gradient
    X_grad, theta_grad = rc.deserialize(rc.regularized_gradient(param, Y, R, num_features),
                                        num_movies, num_users, num_features)

    # Parse movie_id.txt
    movie_list = []
    with open('./data/movie_ids.txt', encoding='latin-1') as file:
        for l in file:
            tokens = l.strip().split(' ')
            movie_list.append(' '.join(tokens[1:]))

    movie_list = np.array(movie_list)

    ratings = np.zeros(num_movies)
    ratings[0] = 4
    ratings[6] = 3
    ratings[11] = 5
    ratings[53] = 4
    ratings[63] = 5
    ratings[65] = 3
    ratings[68] = 5
    ratings[97] = 2
    ratings[182] = 4
    ratings[225] = 5
    ratings[354] = 5

    Y = np.insert(Y, 0, ratings, axis=1)
    R = np.insert(R, 0, ratings != 0, axis=1)

    num_features = 50
    num_movies, num_users = Y.shape
    l = 10

    X = np.random.standard_normal((num_movies, num_features))
    theta = np.random.standard_normal((num_users, num_features))

    param = rc.serialize(X, theta)

    Y_norm = Y - Y.mean()

    res = opt.minimize(fun=rc.regularized_cost,
                       x0=param,
                       args=(Y_norm, R, num_features, l),
                       method='TNC',
                       jac=rc.regularized_gradient)

    X_trained, theta_trained = rc.deserialize(res.x, num_movies, num_users, num_features)
    prediction = X_trained @ theta_trained.T
    my_preds = prediction[:, 0] + Y.mean()
    idx = np.argsort(my_preds)[::-1]
    for m in movie_list[idx][:10]:
        print(m)


if __name__ == '__main__':
    main()