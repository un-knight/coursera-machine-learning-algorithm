"""
author: Junxian Ye
time: 01/17/2017
link: https://github.com/un-knight/coursera-machine-learning-algorithm
"""

import scipy.io as sio
import skimage.io as img_io
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt


def random_init(data, k):
    m = data.sample(k).as_matrix()
    return m

def main():
    # visualize the data
    """
    mat = sio.loadmat("./data/ex7data2.mat")
    data = pd.DataFrame(mat.get("X"), columns=['X1', 'X2'])
    # print("data2: \n", data.head())
    # sns.lmplot('X1', 'X2', data=data, fit_reg=False)
    # plt.show()

    km = KMeans(n_clusters=3)
    km.fit(data)
    result = km.predict(data)
    # print(result)
    data_result = data.copy()
    data_result['kind'] = result
    print(data_result.head())
    sns.lmplot('X1', 'X2', hue='kind', data=data_result, fit_reg=False)
    plt.title("k-means with sklearn")
    plt.show()
    """

    # Image compression
    img = img_io.imread("./data/bird_small.png") / 255
    # img_io.imshow(img)
    # plt.show()
    # print(img.shape)

    img_data = img.reshape(128*128, 3)
    # speed up computation by setting n_jobs=-1 to use all CPU kernels
    km = KMeans(n_clusters=16, n_jobs=-1)
    km.fit(img_data)
    centroids = km.cluster_centers_
    # print(centroids, centroids.shape)

    result = km.predict(img_data)
    compressed_img = centroids[result].reshape(128, 128, 3)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img)
    ax[1].imshow(compressed_img)
    plt.show()


if __name__ == "__main__":
    main()