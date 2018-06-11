from utils import data_helpers
from sklearn import datasets
from sknn.mlp import Layer, MultiLayerPerceptron, Classifier
import random
import numpy as np
from sklearn.neural_network import MLPClassifier


def load_train_data(split_percent=0.9):
    iris = datasets.load_iris()
    feature_data = iris.data
    label_data = iris.target
    shuffle_indices = range(len(feature_data))
    random.shuffle(shuffle_indices)
    shuffled_feature_data = np.array(feature_data)[shuffle_indices]
    shuffled_label_data = np.array(label_data)[shuffle_indices]
    split_index = int(split_percent * len(shuffled_feature_data))
    feature_train = shuffled_feature_data[:split_index]
    label_train = shuffled_label_data[:split_index]
    feature_test = shuffled_feature_data[split_index:]
    label_test = shuffled_label_data[split_index:]
    return feature_train, label_train, feature_test, label_test


def convert_label(labels):
    new_labels = []
    for label_ in labels:
        if label_ == 0:
            new_labels.append([1, 0, 0])
        elif label_ == 1:
            new_labels.append([0, 1, 0])
        else:
            new_labels.append([0, 0, 1])
    return new_labels


def train_sknn(data, labels):
    # layer one:
    hidden_layer = Layer(type="Sigmoid", name="hidden", units=10)
    output_layer = Layer(type="Softmax", name="output")
    layers = [hidden_layer, output_layer]
    mlp = Classifier(layers=layers, random_state=1)
    mlp.fit(data, labels)
    return mlp


def train(data, labels):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 5), random_state=1)
    clf.fit(data, labels)
    return clf


def test(data, labels, model):
    predict_labels = model.predict(data)
    total = 0
    count = 0
    for id_, label_ in enumerate(labels):
        if max(label_) == max(predict_labels[id_]):
            count += 1
        total += 1
    print('accuracy %f' % (float(count) / total))


def main():
    print('start')
    train_data, train_labels, test_data, test_labels = load_train_data()
    train_labels = convert_label(train_labels)
    test_labels = convert_label(test_labels)
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)
    model = train(train_data, train_labels)
    test(test_data, test_labels, model)


if __name__ == '__main__':
    main()

