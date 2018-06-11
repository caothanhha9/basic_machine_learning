import data_processing
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from matplotlib import pyplot as plt


x, y = data_processing.load_car_price('car_price2.csv')
x_train = x[:120]
y_train = y[:120]

X = np.expand_dims(x, -1)
X_train = np.expand_dims(x_train, -1)
colors = ['red', 'blue', 'green']

plt.scatter(X, y, color='black', label='real')

for count, degree in enumerate([2, 3, 4]):
    model = make_pipeline(PolynomialFeatures(degree), Ridge())
    # model = PolynomialFeatures(degree)
    model.fit(X_train, y_train)
    y_predict = model.predict(X)
    plt.plot(X, y_predict, color=colors[count], label="degree {}".format(degree))

plt.legend(loc='lower left')
plt.xlabel('Miles')
plt.ylabel('Price ($)')
plt.show()



