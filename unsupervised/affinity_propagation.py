from sklearn.cluster import AffinityPropagation
from utils import data_helpers
import matplotlib.pyplot as plt


def customer_clustering():
    data = data_helpers.read_feature_data(file_path='../data/customer_data_min')
    aff_prop_model = AffinityPropagation(convergence_iter=150, max_iter=1000)
    aff_prop_model.fit(data)
    cluster_centers_indices = aff_prop_model.cluster_centers_indices_
    labels = aff_prop_model.labels_
    print(labels)
    return data, labels


def plot(data, labels):
    xs0, ys0, zs0 = data_helpers.get_data_by_label(data, labels, 0)
    xs1, ys1, zs1 = data_helpers.get_data_by_label(data, labels, 1)
    xs2, ys2, zs2 = data_helpers.get_data_by_label(data, labels, 2)
    xs3, ys3, zs3 = data_helpers.get_data_by_label(data, labels, 3)
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs0, zs0, ys0, c='y', marker='o')
    ax.scatter(xs1, zs1, ys1, c='r', marker='o')
    ax.scatter(xs2, zs2, ys2, c='b', marker='o')
    ax.scatter(xs3, zs3, ys3, c='g', marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()


def main():
    data, labels = customer_clustering()
    plot(data, labels)


if __name__ == '__main__':
    main()
