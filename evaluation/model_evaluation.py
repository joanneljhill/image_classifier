from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as mp
from model_training import *

def evaluate(trained_model, X_test, y_test):
    """
    Evaluate a model, printing weighted f1 score, f1 score for each class, confusion matrix and score

    :param trained_model: pickle of trained model
    :param X_test: X test data to use to evaluate the model
    :param y_test: y test data to use to evaluate the model
    """

    clf = pickle.load(open(trained_model))

    y_pred = clf.predict(X_test)

    print "Weighted F1 score = %f" %f1_score(y_test, y_pred, pos_label=None, average='weighted')
    print "F1 score = "
    print f1_score(y_test, y_pred, average=None)

    print "Confusion matrix = "
    print confusion_matrix(y_test, y_pred)

    print "Score = %f" %clf.score(X_test, y_test)

def evaluate_n_estimators(X_train, X_test, y_train, y_test):
    """
    Evaluate the model varying the number of n_estimators and plot the results

    :param X_train: X training data
    :param X_test: X test data
    :param y_train: y training data
    :param y_test:  y test data
    """

    estimators = [1, 10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    scores = []

    for value in estimators:
        clf = train_random_forest(X_train, y_train, value, (str(value) + '.p'))
        scores.append(clf.score(X_test, y_test))

    #plot data
    mp.scatter(estimators,scores,color='blue')
    mp.ylabel('n_estimators')
    mp.xlabel('scores')
    mp.savefig('estimators.png')