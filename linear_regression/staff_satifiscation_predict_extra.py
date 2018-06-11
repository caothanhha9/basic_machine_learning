from utils import data_helpers
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt


def load_train_data(file_path='../data/customer_off_time_salary_satisfaction'):
    train_data = data_helpers.read_feature_data(file_path)
    feature_dim = len(train_data[0]) - 1
    feature_data = []
    target_data = []
    for data_sample in train_data:
        feature_vec = []
        for id_ in range(feature_dim):
            feature_vec.append(data_sample[id_])
        feature_data.append(feature_vec)
        target_data.append(data_sample[feature_dim])
    return feature_data, target_data


def train():
    x_data, y_data = load_train_data()
    x_train_data = x_data[:-200]
    y_train_data = y_data[:-200]
    lin_res = linear_model.LinearRegression()
    lin_res.fit(x_train_data, y_train_data)
    print('Coefficients:')
    print(lin_res.coef_)
    return lin_res


def test():
    x_data, y_data = load_train_data()
    x_test_data = x_data[-200:]
    y_test_data = y_data[-200:]
    lin_res = train()
    predict_y_data = lin_res.predict(x_test_data)
    # Mean square error
    mean_sq_err = np.mean((predict_y_data - y_test_data) ** 2)
    print('Mean square errors')
    print(mean_sq_err)
    variance_score = lin_res.score(x_test_data, y_test_data)
    print('Variance score (1.0 is perfect)')
    print(variance_score)
    # Plot data
    # Plot outputs
    x_1_data = np.array(x_test_data)[:, 0]
    x_2_data = np.array(x_test_data)[:, 1]
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_1_data, x_2_data, y_test_data, c='r', marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    ax.scatter(x_1_data, x_2_data, predict_y_data, c='b', marker='D')
    # ax.plot(x_1_data, x_2_data, predict_y_data, c='b', marker='D', linewidth=3)
    # plt.scatter(x_test_data, y_test_data,  color='black')
    # plt.plot(x_test_data, predict_y_data, color='blue',
    #          linewidth=3)
    # plt.xticks(())
    # plt.yticks(())

    plt.show()

if __name__ == '__main__':
    # train()
    test()

