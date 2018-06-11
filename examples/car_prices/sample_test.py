from data_processing import load_car_price
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")


x, y = load_car_price('car_price.csv')
x = np.expand_dims(x, axis=-1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=2)

# Linear regression
reg = LinearRegression()

reg.fit(x_train, y_train)
score = reg.score(x_test, y_test)
print('score: {}'.format(score))
