from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


# load data
data = datasets.load_iris()
# print(data)

features = data.data
labels = data.target

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.10, random_state=2)

# Create model and train
model = GaussianNB()
model.fit(X_train, y_train)

# Test
y_predict = model.predict(X_test)
correct_prediction = [1 if (y_predict[i] == y_test[i]) else 0 for i in range(0, y_test.shape[0])]

acc = float(sum(correct_prediction)) / len(correct_prediction)

print('Accuracy {}'.format(acc))
