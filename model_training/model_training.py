import pickle
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

def train_random_forest(X_train, y_train, estimators, model_name):
    """
    Trains a random forest model and stores the trained model as a pickle

    :param X_train: X training data
    :param y_train: y training data
    """

    clf = RandomForestClassifier(n_estimators=estimators)
    clf = clf.fit(X_train, y_train)
    pickle.dump(clf, open(model_name,'w'))

def split_data(X, y, test_size):
    """
    Splits X and y training data into a test and training set

    :param X: all X data
    :param y: all y data
    :param test_size: size of the test data set
    :return: training and test data - X_train, X_test, y_train, y_test
    """

    return cross_validation.train_test_split(X, y, test_size=test_size, random_state=42)


