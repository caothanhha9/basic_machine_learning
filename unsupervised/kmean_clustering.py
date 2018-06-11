from sklearn import datasets
from sklearn.cluster import KMeans
from utils import data_helpers
import matplotlib.pyplot as plt


def iris_clustering():
    iris = datasets.load_iris()
    data = iris['data']
    labels = iris['target']
    kmean_model = KMeans(n_clusters=3)
    kmean_model.fit(data)
    print(kmean_model)
    centroids = kmean_model.cluster_centers_
    print('centers:')
    print(centroids)
    predict_labels = kmean_model.predict(data)
    print(predict_labels)
    total_guess_num = 150
    correct_guess_num = 0
    predict_label_count_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for id_, label_ in enumerate(predict_labels):
        if id_ < 50:
            predict_label_count_matrix[0][label_] += 1
        elif id_ < 100:
            predict_label_count_matrix[1][label_] += 1
        else:
            predict_label_count_matrix[2][label_] += 1
    for i in range(3):
        correct_guess_num += max(predict_label_count_matrix[i])

    accuracy = float(correct_guess_num) / float(total_guess_num)
    print('accuracy:' + str(accuracy))
    return data, predict_labels


def iris_plot(data, labels):
    xs0, ys0, zs0 = data_helpers.get_data_by_label(data, labels, 0)
    xs1, ys1, zs1 = data_helpers.get_data_by_label(data, labels, 1)
    xs2, ys2, zs2 = data_helpers.get_data_by_label(data, labels, 2)
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs0, zs0, ys0, c='y', marker='o')
    ax.scatter(xs1, zs1, ys1, c='r', marker='o')
    ax.scatter(xs2, zs2, ys2, c='b', marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()


def customer_clustering():
    data = data_helpers.read_feature_data(file_path='../data/customer_data')
    kmean_model = KMeans(n_clusters=4)
    predict_labels = kmean_model.fit_predict(data)
    cluster_centers_indices = kmean_model.cluster_centers_
    print(predict_labels)
    total_guess_num = 800
    correct_guess_num = 0
    predict_label_count_matrix = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    for id_, label_ in enumerate(predict_labels):
        if id_ < 200:
            predict_label_count_matrix[0][label_] += 1
        elif id_ < 400:
            predict_label_count_matrix[1][label_] += 1
        elif id_ < 600:
            predict_label_count_matrix[2][label_] += 1
        else:
            predict_label_count_matrix[3][label_] += 1
    for i in range(4):
        correct_guess_num += max(predict_label_count_matrix[i])

    accuracy = float(correct_guess_num) / float(total_guess_num)
    print('accuracy:' + str(accuracy))
    return data, predict_labels


def customer_plot(data, labels):
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
    print('starting...')
    data, labels = iris_clustering()
    iris_plot(data, labels)
    # data, labels = customer_clustering()
    # customer_plot(data, labels)


if __name__ == '__main__':
    main()


