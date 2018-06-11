import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn import decomposition
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")


data = pd.read_csv("/Users/admin/Desktop/Projects/python/machine_learning/examples/housing_prices/kc_house_data.csv")

conv_dates = [1 if values == 2014 else 0 for values in data.date ]
data['date'] = conv_dates
X_train = data.drop(['id', 'price'], axis=1)
labels = data['price']
x_train, x_test, y_train, y_test = train_test_split(X_train, labels, test_size=0.10, random_state=2)


def test_linear():
    # Linear regression
    reg = LinearRegression()

    reg.fit(x_train, y_train)
    score = reg.score(x_test, y_test)
    print('linear score: {}'.format(score))


def test_pca_linear():
    # PCA
    scaler = MinMaxScaler()
    scaler.fit(X_train)

    X_norm = scaler.transform(X_train)

    """n_components can be integer (number of features) or float (ratio of info cap)"""
    pca = decomposition.PCA(n_components=10)
    pca.fit(X_norm)

    X_reduce = pca.transform(X_norm)
    xr_train, xr_test, yr_train, yr_test = train_test_split(X_reduce, labels, test_size=0.10, random_state=2)
    reg2 = LinearRegression()
    reg2.fit(xr_train, yr_train)
    score2 = reg2.score(xr_test, yr_test)
    print('pca + linear score: {}'.format(score2))


def test_poly():
    model = make_pipeline(PolynomialFeatures(2), Ridge())
    # model = PolynomialFeatures(degree)
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    print('poly score: {}'.format(score))


def test_pca_poly():
    # PCA
    scaler = MinMaxScaler()
    scaler.fit(X_train)

    X_norm = scaler.transform(X_train)

    pca = decomposition.PCA(n_components=10)
    pca.fit(X_norm)

    X_reduce = pca.transform(X_norm)
    xr_train, xr_test, yr_train, yr_test = train_test_split(X_reduce, labels, test_size=0.10, random_state=2)

    model = make_pipeline(PolynomialFeatures(2), Ridge())
    model.fit(xr_train, yr_train)
    score2 = model.score(xr_test, yr_test)
    print('pca + poly score: {}'.format(score2))


def test_gradient_boosting():
    # Gradient Boosting Regression
    from sklearn import ensemble
    clf = ensemble.GradientBoostingRegressor(n_estimators=400, max_depth=5, min_samples_split=2,
                                             learning_rate=0.1, loss='ls')

    clf.fit(x_train, y_train)
    gr_score = clf.score(x_test, y_test)
    print('gradient boosting score: {}'.format(gr_score))


def test_pca_gradient_boosting():
    # PCA
    scaler = MinMaxScaler()
    scaler.fit(X_train)

    X_norm = scaler.transform(X_train)

    pca = decomposition.PCA(n_components=10)
    pca.fit(X_norm)

    X_reduce = pca.transform(X_norm)
    xr_train, xr_test, yr_train, yr_test = train_test_split(X_reduce, labels, test_size=0.10, random_state=2)

    from sklearn import ensemble
    model = ensemble.GradientBoostingRegressor(n_estimators=400, max_depth=5, min_samples_split=2,
                                               learning_rate=0.1, loss='ls')
    model.fit(xr_train, yr_train)
    score2 = model.score(xr_test, yr_test)
    print('pca + boosting score: {}'.format(score2))


if __name__ == '__main__':
    # test_linear()
    # test_pca_linear()
    # test_poly()
    # test_pca_poly()
    test_gradient_boosting()
    test_pca_gradient_boosting()
