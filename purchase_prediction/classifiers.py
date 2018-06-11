from sklearn.linear_model import LogisticRegression
from utils import data_helpers
import numpy as np
import random
import pickle


def get_data_by_label(feature_data, label_data, expect_label):
    f_feature_data = []
    f_label_data = []
    o_feature_data = []
    o_label_data = []
    for id_, feature_ in enumerate(feature_data):
        label_ = label_data[id_]
        if label_ == expect_label:
            f_feature_data.append(feature_)
            f_label_data.append(label_)
        else:
            o_feature_data.append(feature_)
            o_label_data.append(label_)
    return f_feature_data, f_label_data, o_feature_data, o_label_data


def balance_data(feature_data, label_data):
    feature_data0, label_data0, feature_data1, label_data1 = get_data_by_label(
        feature_data, label_data, 0)
    min_len = min([len(feature_data0), len(feature_data1)])
    b_feature_data = feature_data0[:min_len] + feature_data1[:min_len]
    b_label_data = label_data0[:min_len] + label_data1[:min_len]
    return b_feature_data, b_label_data


def load_data(file_path='../data/purchase_data_20000.processed', split_percent=0.8):
    o_feature_data, o_label_data = data_helpers.read_feature_label_data(file_path)
    feature_data, label_data = balance_data(o_feature_data, o_label_data)
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


def train(feature_data, label_data):
    lg_model = LogisticRegression()
    lg_model.fit(feature_data, label_data)
    # saved_model = pickle.dumps(model)
    saved_path = open('./model.pkl', 'wb')
    pickle.dump(lg_model, saved_path)
    saved_path.close()
    return lg_model


def test(feature_data, label_data):
    saved_path = open('./model.pkl', 'rb')
    saved_model = pickle.load(saved_path)
    predicted_label = saved_model.predict(feature_data)
    total_num = 0
    correct_num = 0
    for id_, label_ in enumerate(predicted_label):
        print("-" * 40)
        print(label_)
        print(label_data[id_])
        if label_ == label_data[id_]:
            correct_num += 1
        total_num += 1
    accuracy = float(correct_num) / float(total_num)
    print(accuracy)


def main():
    print('start')
    feature_train, label_train, feature_test, label_test = load_data()
    train(feature_train, label_train)
    test(feature_test, label_test)


if __name__ == "__main__":
    main()

