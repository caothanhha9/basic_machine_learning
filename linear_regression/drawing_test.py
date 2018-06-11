# -*- coding: utf-8 -*-
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def get_diabete_data():
    """
    Hầm được sử dụng để load dữ liệu diabete từ thư viện sklearn
    Sử dụng module datasets trong thư viện sklearn, hàm load_tên_bộ_dữ_liệu() được sử
    dụng để load bộ dữ liệu tương ứng. Dữ liệu trả về dưới dạng dictionary.
    Ta sử dụng key là 'data' để lấy ra dữ liệu dạng số.
    :return:
    """
    diabete_load = datasets.load_diabetes()
    return diabete_load['data']


def plot_data():
    xs = range(0, 100)
    ys = map(lambda x: pow(x, 2), xs)
    plt.plot(xs, ys, c='r', marker='o')
    plt.show()
    return True


def scatter_data(feature_size=3):
    diabete_data = np.array(get_diabete_data())
    xs = diabete_data[:, 0]
    ys = diabete_data[:, 1]
    if feature_size == 2:
        plt.scatter(xs, ys, c='b', marker='o')
    elif feature_size == 3:
        zs = diabete_data[:, 2]
        fig = plt.figure()

        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xs, zs, ys, c='r', marker='o')

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
    plt.show()
    return True

if __name__ == '__main__':
    # Load dữ liệu từ sklearn sử dụng hàm đc định nghĩa phía trên
    diabete_data = get_diabete_data()
    # In ra số mẫu (sample) của bộ dữ liệu diabete
    print('Số lượng mẫu')
    print(len(diabete_data))
    print('Kích thước không gian feature')
    print(len(diabete_data[0]))  # Chiều đầu tiên là label
    plot_data()
    scatter_data()
