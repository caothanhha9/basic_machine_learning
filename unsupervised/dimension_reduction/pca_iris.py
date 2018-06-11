import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from sklearn import decomposition
from sklearn import datasets


def test_3d():
    centers = [[1, 1], [-1, -1], [1, -1]]
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    plt.cla()
    pca = decomposition.PCA(n_components=3)
    pca.fit(X)
    X = pca.transform(X)

    for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
        ax.text3D(X[y == label, 0].mean(),
                  X[y == label, 1].mean() + 1.5,
                  X[y == label, 2].mean(), name,
                  horizontalalignment='center',
                  bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
    # Reorder the labels to have colors matching the cluster results
    y = np.choose(y, [1, 2, 0]).astype(np.float)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.spectral,
               edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])

    plt.show()


def show_origin():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    show_iris_2d(X, y, select_index=(0, 1))


def show_iris_2d(X, y, select_index=(0, 1)):
    setosa_list = []
    versicolor_list = []
    virginica_list = []
    for i in range(0, len(X)):
        if y[i] == 0:
            setosa_list.append(X[i])
        elif y[i] == 1:
            versicolor_list.append(X[i])
        else:
            virginica_list.append(X[i])
    setosa_list = np.asarray(setosa_list)
    versicolor_list = np.asarray(versicolor_list)
    virginica_list = np.asarray(virginica_list)

    plt.scatter(setosa_list[:, select_index[0]], setosa_list[:, select_index[1]], color='red')
    plt.scatter(versicolor_list[:, select_index[0]], versicolor_list[:, select_index[1]], color='blue')
    plt.scatter(virginica_list[:, select_index[0]], virginica_list[:, select_index[1]], color='green')

    plt.show()


def test_2d():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    pca = decomposition.PCA(n_components=2)
    pca.fit(X)
    X = pca.transform(X)
    show_iris_2d(X, y, select_index=(0, 1))

if __name__ == '__main__':
    show_origin()
    test_2d()


