from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn import svm
from sklearn import datasets

iris = datasets.load_iris()

clf = svm.SVC(kernel='linear', C=1)

shuffle_option = 1
if shuffle_option == 0:
    cv = 5
else:
    cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
scores = cross_val_score(clf, iris.data, iris.target, cv=cv)
print(scores)
