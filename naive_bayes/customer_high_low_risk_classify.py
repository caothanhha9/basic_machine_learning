from sklearn.naive_bayes import GaussianNB
from utils import data_helpers
import random
import numpy as np
import pickle
import matplotlib.pyplot as plt


def load_train_data(split_percent=0.8):
    feature_data, label_data = data_helpers.read_feature_label_data(file_path='../data/customer_saving_salary')
    shuffle_indices = range(len(feature_data))
    random.shuffle(shuffle_indices)
    shuffled_feature_data = np.array(feature_data)[shuffle_indices]
    shuffled_label_data = np.array(label_data)[shuffle_indices]
    split_index = int(split_percent * len(shuffled_feature_data))
    feature_train = shuffled_feature_data[:-split_index]
    label_train = shuffled_label_data[:-split_index]
    feature_test = shuffled_feature_data[-split_index:]
    label_test = shuffled_label_data[-split_index:]
    return feature_train, label_train, feature_test, label_test


def train():
    feature_data, label_data, _, _ = load_train_data()
    gnb = GaussianNB()
    model = gnb.fit(feature_data, label_data)
    # saved_model = pickle.dumps(model)
    saved_path = open('./model.pkl', 'wb')
    pickle.dump(model, saved_path)
    saved_path.close()
    return model


def test():
    _, _, feature_data, label_data = load_train_data()
    saved_path = open('./model.pkl', 'rb')
    saved_model = pickle.load(saved_path)
    predicted_label = saved_model.predict(feature_data)
    total_num = 0
    correct_num = 0
    for id_, label_ in enumerate(predicted_label):
        if label_ == label_data[id_]:
            correct_num += 1
        total_num += 1
    accuracy = float(correct_num) / float(total_num)
    print(accuracy)


def plot():
    min_num=0.0
    max_num=1.0
    n = 100
    m = 100
    delta_n = (max_num - min_num) / n
    delta_m = (max_num - min_num) / m
    grid_data = []
    for i in range(n):
        for j in range(m):
            grid_data.append([i * delta_n, j * delta_m])
    saved_path = open('./model.pkl', 'rb')
    saved_model = pickle.load(saved_path)
    predicted_label = saved_model.predict(grid_data)
    xs0 = []
    ys0 = []
    xs1 = []
    ys1 = []
    for point_id, point in enumerate(grid_data):
        if predicted_label[point_id] == 0.0:
            xs0.append(point[0])
            ys0.append(point[1])
        else:
            xs1.append(point[0])
            ys1.append(point[1])
    plt.scatter(xs0, ys0, c='r')
    plt.scatter(xs1, ys1, c='b')
    plt.show()


def main():
    print('starting...')
    # train()
    # test()
    plot()


if __name__ == '__main__':
    main()
