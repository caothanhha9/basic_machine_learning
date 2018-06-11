from sklearn.datasets import load_iris
from sklearn import tree
import graphviz
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# NOTICE: pip install graphviz


def get_data_range(data, start_index, end_index, x_id, y_id, z_id):
    xs = np.array(data)[start_index:end_index, x_id]
    ys = np.array(data)[start_index:end_index, y_id]
    zs = np.array(data)[start_index:end_index, z_id]
    return xs, ys, zs


def plot_iris_data():
    feature_data = load_iris()["data"]
    label_data = load_iris()["target"]
    xs1, ys1, zs1 = get_data_range(feature_data, 0, 50, 1, 2, 3)
    xs2, ys2, zs2 = get_data_range(feature_data, 50, 100, 1, 2, 3)
    xs3, ys3, zs3 = get_data_range(feature_data, 100, 150, 1, 2, 3)
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs1, zs1, ys1, c='r', marker='o')
    ax.scatter(xs2, zs2, ys2, c='b', marker='o')
    ax.scatter(xs3, zs3, ys3, c='g', marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()


def test_iris():
    iris = load_iris()
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(iris.data, iris.target)

    dot_data = tree.export_graphviz(clf, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render("iris")


def test_iris2():
    iris = load_iris()
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(iris.data, iris.target)

    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=iris.feature_names,
                                    class_names=iris.target_names,
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("iris")


def load_car_price(file_path):
    column_names = ['mileage', 'price']
    df = pd.read_csv(file_path, header=0, names=column_names)
    x = df['mileage'].as_matrix()
    y = df['price'].as_matrix()
    return x, y


def test_regression_car_price():
    x, y = load_car_price('car_price.csv')
    x = np.expand_dims(x, axis=-1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=2)
    regressor = DecisionTreeRegressor(random_state=0)
    regressor.fit(x_train, y_train)
    score = regressor.score(x_test, y_test)
    print(score)


def test_loan_approval():
    train = pd.read_csv("loan_train.csv")

    def process_data(df):
        df['Property_Area'].fillna('missing')
        df['Gender'].fillna('missing')
        df['Married'].fillna('missing')
        df['Education'].fillna('missing')
        df['Self_Employed'].fillna('missing')
        df['Dependents'].fillna('missing')

        df['Property_Area'] = df['Property_Area'].astype('category')
        df['Gender'] = df['Gender'].astype('category')
        df['Married'] = df['Married'].astype('category')
        df['Education'] = df['Education'].astype('category')
        df['Self_Employed'] = df['Self_Employed'].astype('category')
        df['Dependents'] = df['Dependents'].astype('category')

        df['Loan_Status'] = df['Loan_Status'].astype('category')

        cat_columns = df.select_dtypes(['category']).columns
        df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)

        df = df[np.isfinite(df['LoanAmount'])]
        df = df[np.isfinite(df['Loan_Amount_Term'])]
        df = df[np.isfinite(df['Credit_History'])]
        df = df[np.isfinite(df['ApplicantIncome'])]
        df = df[np.isfinite(df['CoapplicantIncome'])]
        return df

    train = process_data(train)

    X_data = train.drop(['Loan_ID', 'Loan_Status'], axis=1)
    y_data = train['Loan_Status']

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.10, random_state=2)

    # Train and test
    model = tree.DecisionTreeClassifier()
    model.fit(X_train, y_train)

    y_predict = model.predict(X_test)

    y_true = y_test.as_matrix()
    print(y_true.shape[0])

    correct_prediction = [1 if (y_true[i] == y_predict[i]) else 0 for i in range(0, y_true.shape[0])]
    acc = float(np.sum(correct_prediction)) / len(correct_prediction)
    print('accuracy {}'.format(acc))


def test_load_approval_multiple():
    for _ in range(10):
        test_loan_approval()


if __name__ == '__main__':
    plot_iris_data()
    # test_iris()
    # test_iris2()
    # test_regression_car_price()
    # test_load_approval_multiple()
