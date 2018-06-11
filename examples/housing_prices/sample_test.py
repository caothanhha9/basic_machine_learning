import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")


data = pd.read_csv("kc_house_data.csv")

conv_dates = [1 if values == 2014 else 0 for values in data.date ]
data['date'] = conv_dates
X_train = data.drop(['id', 'price'], axis=1)
labels = data['price']


# Linear regression
reg = LinearRegression()

x_train, x_test, y_train, y_test = train_test_split(X_train, labels, test_size=0.10, random_state=2)
reg.fit(x_train, y_train)
score = reg.score(x_test, y_test)
print('score: {}'.format(score))

# Gradient Boosting Regression
from sklearn import ensemble
clf = ensemble.GradientBoostingRegressor(n_estimators=400, max_depth=5, min_samples_split=2,
                                         learning_rate=0.1, loss='ls')

clf.fit(x_train, y_train)
gr_score = clf.score(x_test, y_test)
print('gradient boosting score: {}'.format(gr_score))

