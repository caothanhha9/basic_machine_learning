from sklearn.linear_model import Perceptron


def and_function(b1, b2):
    data = [[0, 0], [0, 1], [1, 0], [1, 1]]
    labels = [0, 0, 0, 1]
    perceptron = Perceptron()
    perceptron.fit(data, labels)
    y = perceptron.predict([[b1, b2]])
    return y[0]


def main():
    print('start')
    print and_function(1, 1)


if __name__ == '__main__':
    main()
