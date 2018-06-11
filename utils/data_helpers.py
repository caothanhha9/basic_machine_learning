import os
import random
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def auto_gen_2d_regression_data(n=100, a=1.0, b=0.0, delta_b=0.05, x_range=[0.0, 1.0]):
    n = int(n)
    data = []
    if n > 0:
        for i in range(n):
            x_ = float(i) * (x_range[1] - x_range[0]) / n
            y_ = a * x_ + b + random.uniform(-1.0 * delta_b, delta_b)
            data.append([x_, y_])
    return data


def auto_gen_3d_regression_data(n=100, a=1.0, b=1.0, c=0.0, delta_c=0.05, x_range=[0.0, 1.0], y_range=[0.0, 1.0]):
    n = int(n)
    data = []
    if n > 0:
        for i in range(n):
            x_ = float(i) * (x_range[1] - x_range[0]) / n
            for j in range(n):
                y_ = float(j) * (y_range[1] - y_range[0]) / n
                z_ = a * x_ + b * y_ + random.uniform(-1.0 * delta_c, delta_c)
                data.append([x_, y_, z_])
    return data


def auto_gen_2d_classification_data(n=100, min_num=0.0, max_num=1.0):
    data = []
    avg_num = (min_num + max_num) / 2.0
    for id_ in range(n):
        ft1 = random.uniform(min_num, max_num)
        ft2 = random.uniform(min_num, max_num)
        if (ft1 < avg_num) & (ft2 < avg_num):
            label = 0
            data.append([ft1, ft2, label])
        elif (ft1 > avg_num) & (ft2 > avg_num):
            label = 1
            data.append([ft1, ft2, label])
    return data


def distance(vector_a, vector_b):
    size_a = size_b = len(vector_a)
    dist = 0.0
    for i in range(size_a):
        dist += math.pow(vector_a[i] - vector_b[i], 2)
    dist = math.sqrt(dist)
    return dist


def auto_gen_3d_clusters(centroids, n_samples=100, radius_ratio=0.95):
    data = []
    n_clusters = len(centroids)
    min_dist = 1.0e10
    for i in range(n_clusters):
        other_ids = list(set(range(n_clusters)) - {i})
        distances = []
        for o_i in other_ids:
            distances.append(distance(centroids[i], centroids[o_i]))
        min_dist_ = min(distances)
        if min_dist_ < min_dist:
            min_dist = min_dist_
    for i in range(n_clusters):
        # other_ids = list(set(range(n_clusters)) - {i})
        # distances = []
        # for o_i in other_ids:
        #     distances.append(distance(centroids[i], centroids[o_i]))
        # min_dist = min(distances)
        count = 0
        while count < n_samples:
            replace_dist_x = random.uniform(-0.6, 0.6) * min_dist
            replace_dist_y = random.uniform(-0.6, 0.6) * min_dist
            replace_dist_z = random.uniform(-0.6, 0.6) * min_dist
            point = [replace_dist_x, replace_dist_y, replace_dist_z]
            point = map(sum, zip(centroids[i], point))
            real_dist = distance(point, centroids[i])
            if real_dist < (radius_ratio * min_dist * 0.5):
                data.append(point)
                count += 1
    return data


def auto_gen_and_save_cluster_data(file_path, n=100, radius_ratio=0.95):
    centroids = [[0.1, 0.1, 0.1], [0.2, 0.3, 0.2], [0.4, 0.5, 0.4], [0.7, 0.8, 0.7]]
    data = auto_gen_3d_clusters(centroids=centroids, n_samples=n, radius_ratio=radius_ratio)
    f = open(file_path, 'w')
    for sample_ in data:
        sample_str = map(str, sample_)
        f.write('\t'.join(sample_str))
        f.write('\n')
    f.close()


def auto_gen_and_save_classification_data(n=100, file_path=''):
    if (n > 0) & (len(file_path) > 0):
        data = auto_gen_2d_classification_data(n=n)
        f = open(file_path, 'w')
        for sample_ in data:
            sample_str = map(str, sample_)
            f.write('\t'.join(sample_str))
            f.write('\n')
        f.close()


def auto_gen_and_save_regression_data(n=100, file_path='', feature_dim=2):
    if (n > 0) & (len(file_path) > 0):
        data = None
        if feature_dim == 2:
            data = auto_gen_2d_regression_data(n=n)
        if feature_dim == 3:
            data = auto_gen_3d_regression_data(n=n)
        if data is not None:
            f = open(file_path, 'w')
            for sample_ in data:
                sample_str = map(str, sample_)
                f.write('\t'.join(sample_str))
                f.write('\n')
            f.close()


def read_data(file_path):
    data = []
    if os.path.isfile(file_path):
        f = open(file_path, 'r')
        samples = list(f.readlines())
        f.close()
        samples = [sample.strip() for sample in samples if len(sample.strip()) > 0]
        for sample in samples:
            sample_arr = sample.split('\t')
            feature_arr = map(float, sample_arr[:-1])
            label = int(sample_arr[-1])
            data.append((feature_arr, label))
    return data


def read_feature_data(file_path):
    data = []
    if os.path.isfile(file_path):
        f = open(file_path, 'r')
        samples = list(f.readlines())
        f.close()
        samples = [sample.strip() for sample in samples if len(sample.strip()) > 0]
        for sample in samples:
            sample_arr = sample.split('\t')
            feature_arr = map(float, sample_arr)
            data.append(feature_arr)
    return data


def read_feature_label_data(file_path):
    feature_data = []
    target_data = []
    if os.path.isfile(file_path):
        f = open(file_path, 'r')
        samples = list(f.readlines())
        f.close()
        samples = [sample.strip() for sample in samples if len(sample.strip()) > 0]
        for sample in samples:
            sample_arr = sample.split('\t')
            feature_label_arr = map(float, sample_arr)
            feature_arr = feature_label_arr[:-1]
            feature_data.append(feature_arr)
            target_data.append(int(feature_label_arr[-1]))
    return feature_data, target_data


def plot_data(file_path, plot_label=False):
    data = read_data(file_path)
    if len(data) > 0:
        xs = []
        ys = []
        labels = []
        for sample in data:
            feature_arr = sample[0]
            label = sample[1]
            xs.append(feature_arr[0])
            ys.append(feature_arr[1])
            labels.append(label)
        if plot_label:
            fig = plt.figure()

            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(xs, ys, labels, c='r', marker='o')

            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
        else:
            plt.scatter(xs, ys, c='b', marker='o')

        plt.show()


def plot_regression_data(file_path):
    data = read_feature_data(file_path)
    if len(data) > 0:
        feature_size = len(data[0])
        xs = []
        ys = []
        zs = []
        for sample in data:
            xs.append(sample[0])
            if feature_size > 1:
                ys.append(sample[1])
            if feature_size > 2:
                zs.append(sample[2])
        if feature_size >= 3:
            fig = plt.figure()

            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(xs, ys, zs, c='r', marker='o')

            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
        elif feature_size == 2:
            plt.scatter(xs, ys, c='b', marker='o')
        else:
            plt.scatter(xs, c='b', marker='o')

        plt.show()


def get_data_range(data, start_index, end_index, x_id, y_id, z_id):
    xs = np.array(data)[start_index:end_index, x_id]
    ys = np.array(data)[start_index:end_index, y_id]
    zs = np.array(data)[start_index:end_index, z_id]
    return xs, ys, zs


def get_data_by_label(feature_data, label_data, expect_label):
    feature_dim = len(feature_data[0])
    xs = []
    ys = []
    zs = []
    for sample_id, feature_vector in enumerate(feature_data):
        if label_data[sample_id] == expect_label:
            if feature_dim >= 1:
                xs.append(feature_vector[0])
            if feature_dim >= 2:
                ys.append(feature_vector[1])
            if feature_dim >= 3:
                zs.append(feature_vector[2])
    return xs, ys, zs


def main():
    print('starting...')
    # auto_gen_and_save_classification_data(n=1000, file_path='../data/customer_saving_salary')
    # plot_data(file_path='../data/customer_saving_salary')
    # auto_gen_and_save_regression_data(n=1000, file_path='../data/customer_salary_satisfaction')
    # plot_regression_data(file_path='../data/customer_salary_satisfaction')
    # auto_gen_and_save_regression_data(n=100, file_path='../data/customer_off_time_salary_satisfaction',
    #                                   feature_dim=3)
    # plot_regression_data(file_path='../data/customer_off_time_salary_satisfaction')
    # auto_gen_and_save_cluster_data(file_path='../data/customer_data_min', n=50, radius_ratio=0.8)
    # plot_regression_data(file_path='../data/customer_data_min')
    plot_regression_data(file_path='../data/user_movie_rating')

if __name__ == '__main__':
    main()
