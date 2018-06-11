from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split


def iris_classification():
    iris = datasets.load_iris()
    data = iris.data
    labels = iris.target
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.10, random_state=2)
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    predict_labels = lr_model.predict(X_test)
    print(predict_labels)
    params = lr_model.get_params()
    print(params)
    correct_predictions = [1 if (predict_labels[i] == y_test[i]) else 0 for i in range(0, y_test.shape[0])]
    acc = float(sum(correct_predictions)) / len(correct_predictions)
    print('Accuracy {}'.format(acc))


def main():
    print('starting')
    iris_classification()


if __name__ == '__main__':
    main()
