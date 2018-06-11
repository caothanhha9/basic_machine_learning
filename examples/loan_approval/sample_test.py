import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


"""Statistics on data"""

train = pd.read_csv("train.csv")
# test = pd.read_csv("test.csv") NOTICE: This data does NOT have labels :(

# See first ten samples
print('First ten samples')
# print(train.head(10))

# Statistics of each dimension
print('Statistics of each dimension')
# print(train.describe())

# Check number of values
print('Number of values ~ number of samples')
# sum(train["Property_Area"].value_counts())

# Histogram of one field
# plt.figure(figsize=(8, 5))
# train["ApplicantIncome"].hist(bins=50)
# plt.show()

# Plot box of two fields
# train.boxplot(column="ApplicantIncome", by="Property_Area")
# plt.show()

# Check missing values
print('Missing value statistics')
print(train.apply(lambda x: sum(x.isnull()), axis=0))

"""Processing data"""


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

###############################
"""Create model and training"""


def train(model):
    model.fit(X_train, y_train)
    return model

"""Testing"""


def test(model):
    y_predict = model.predict(X_test)

    y_true = y_test.as_matrix()
    print(y_true.shape[0])

    correct_prediction = [1 if (y_true[i] == y_predict[i]) else 0 for i in range(0, y_true.shape[0])]
    acc = float(np.sum(correct_prediction)) / len(correct_prediction)
    print('accuracy {}'.format(acc))


def train_and_test():
    model = GaussianNB()
    model = train(model)
    test(model)

    model_SVM = SVC(kernel='sigmoid')
    model_SVM = train(model_SVM)
    test(model_SVM)


if __name__ == '__main__':
    train_and_test()

