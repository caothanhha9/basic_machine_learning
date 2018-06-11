import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier


# Reference link: https://www.kaggle.com/c/customer-churn-prediction/data
def process_data(df):
    df['st'].fillna('missing')
    df['acclen'].fillna('missing')
    df['arcode'].fillna('missing')
    df['phnum'].fillna('missing')
    df['intplan'].fillna('missing')
    df['voice'].fillna('missing')

    df['st'] = df['st'].astype('category')
    df['acclen'] = df['acclen'].astype('category')
    df['arcode'] = df['arcode'].astype('category')
    df['phnum'] = df['phnum'].astype('category')
    df['intplan'] = df['intplan'].astype('category')
    df['voice'] = df['voice'].astype('category')

    df['label'] = df['label'].astype('category')

    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)

    df = df[np.isfinite(df['nummailmes'])]
    df = df[np.isfinite(df['tdmin'])]
    df = df[np.isfinite(df['tdcal'])]
    df = df[np.isfinite(df['tdchar'])]
    df = df[np.isfinite(df['temin'])]
    df = df[np.isfinite(df['tecal'])]
    df = df[np.isfinite(df['techar'])]
    df = df[np.isfinite(df['tnmin'])]
    df = df[np.isfinite(df['tncal'])]
    df = df[np.isfinite(df['tnchar'])]
    df = df[np.isfinite(df['timin'])]
    df = df[np.isfinite(df['tical'])]
    df = df[np.isfinite(df['tichar'])]
    df = df[np.isfinite(df['ncsc'])]

    return df


# Train and test
train = pd.read_csv("churn_train.txt")

train = process_data(train)

# NOTICE: Try to select features?
# X_data = train.drop(['label'], axis=1)
X_data = train.drop(['label', 'st', 'acclen', 'arcode', 'phnum', 'intplan', 'voice'], axis=1)

y_data = train['label']

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.10, random_state=2)


def test_model(model):
    model.fit(X_train, y_train)

    y_predict = model.predict(X_test)

    y_true = y_test.as_matrix()
    print(y_true.shape[0])

    correct_prediction = [1 if (y_true[i] == y_predict[i]) else 0 for i in range(0, y_true.shape[0])]
    acc = float(np.sum(correct_prediction)) / len(correct_prediction)
    print('accuracy {}'.format(acc))


def decision_tree_test():
    model = tree.DecisionTreeClassifier()
    test_model(model)


def random_forest_test():
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    test_model(clf)


def ensemble_voting_test():
    clf1 = LogisticRegression(random_state=1)
    clf2 = RandomForestClassifier(random_state=1)
    clf3 = GaussianNB()
    eclf3 = VotingClassifier(estimators=[
        ('lr', clf1), ('rf', clf2), ('gnb', clf3)],
        voting='soft', weights=[2, 1, 1],
        flatten_transform=True)
    test_model(eclf3)


def bagging_test():
    bagging = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5,
                                max_features=0.5)
    test_model(bagging)


if __name__ == '__main__':
    # decision_tree_test()
    # random_forest_test()
    # ensemble_voting_test()
    bagging_test()

