import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd


def test_iris():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.10, random_state=2)

    clf1 = LogisticRegression(random_state=1)
    clf2 = RandomForestClassifier(random_state=1)
    clf3 = GaussianNB()

    eclf1 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
    eclf1 = eclf1.fit(X_train, y_train)
    predict_labels = eclf1.predict(X_test)
    acc = accuracy_score(y_test, predict_labels)
    print('accuracy {}'.format(acc))


def test_loan_approval(model):
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

    model.fit(X_train, y_train)

    y_predict = model.predict(X_test)

    y_true = y_test.as_matrix()
    print(y_true.shape[0])

    correct_prediction = [1 if (y_true[i] == y_predict[i]) else 0 for i in range(0, y_true.shape[0])]
    acc = float(np.sum(correct_prediction)) / len(correct_prediction)
    print('accuracy {}'.format(acc))


def test_voting1():
    clf1 = LogisticRegression(random_state=1)
    clf2 = RandomForestClassifier(random_state=1)
    clf3 = GaussianNB()

    eclf1 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
    test_loan_approval(eclf1)


def test_voting2():
    clf1 = LogisticRegression(random_state=1)
    clf2 = RandomForestClassifier(random_state=1)
    clf3 = GaussianNB()
    eclf2 = VotingClassifier(estimators=[
        ('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft')
    test_loan_approval(eclf2)


def test_voting3():
    clf1 = LogisticRegression(random_state=1)
    clf2 = RandomForestClassifier(random_state=1)
    clf3 = GaussianNB()
    eclf3 = VotingClassifier(estimators=[
        ('lr', clf1), ('rf', clf2), ('gnb', clf3)],
        voting='soft', weights=[2, 1, 1],
        flatten_transform=True)
    test_loan_approval(eclf3)


def test_bagging_iris():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.10, random_state=2)
    bagging = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5,
                                max_features=0.5)
    bagging.fit(X_train, y_train)
    predict_labels = bagging.predict(X_test)
    acc = accuracy_score(y_test, predict_labels)
    print('accuracy {}'.format(acc))


def test_bagging_loan():
    bagging = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5,
                                max_features=0.5)
    test_loan_approval(bagging)


if __name__ == '__main__':
    # test_iris()
    # test_voting1()
    # test_voting2()
    # test_voting3()
    test_bagging_iris()
    # test_bagging_loan()
