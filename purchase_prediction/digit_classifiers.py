from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn import svm


def classify(classifier=None):
    digits = datasets.load_digits()
    images = digits.images
    labels = digits.target
    sample_size = len(images)
    # Reshape 3D array (sample_size x width x height) to 2D array (sample_size x (area))
    data = images.reshape((sample_size, -1))  # -1 will let numpy to determine the size automatically
    train_data = data[:sample_size / 2]
    train_labels = labels[:sample_size / 2]
    if classifier is None:
        model = LogisticRegression()
    else:
        model = classifier
    model.fit(train_data, train_labels)
    test_data = data[sample_size / 2:]
    test_labels = labels[sample_size / 2:]
    predict_labels = model.predict(test_data)
    total = 0
    count = 0
    for id_, label_ in enumerate(predict_labels):
        real_label = test_labels[id_]
        print('-' * 50)
        print('real label: %d' % real_label + ' and predict label: %d' % label_)
        if label_ == real_label:
            count += 1
        total += 1
    print("accuracy %f" % (float(count) / total))


def main():
    print('start')
    classfier = svm.SVC(gamma=0.001)  # gamma is Kernel coefficient for rbf/poly/sigmoid
    classify()
    classify(classfier)

if __name__ == "__main__":
    main()

