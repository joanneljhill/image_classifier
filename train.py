import os
import pickle
from PIL import Image
import numpy as np
import matplotlib.pyplot as mp

from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

def plot_scores(X_train, X_test, y_train, y_test):
    estimators = [1, 10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    scores = []

    for value in estimators:
        clf = RandomForestClassifier(n_estimators=value)
        clf = clf.fit(X_train, y_train)
        scores.append(clf.score(X_test, y_test))

    #plot data
    mp.scatter(estimators,scores,color='blue')
    mp.ylabel('n_estimators')
    mp.xlabel('scores')
    mp.savefig('estimators.png')

def get_featurevector(image):
    img = Image.open(image).convert('L')
    new_img = img.resize((60,30))

    return np.array(new_img).flatten()

def get_training_data(training_data):
    X = []
    y = []

    for directory in os.listdir(training_data):
        if os.path.isdir(training_data + directory + '/'):
            for file in os.listdir(training_data + directory + '/'):
                if not file.startswith('.'):

                    featurevector = get_featurevector(training_data + directory + '/' + file)

                    X.append(featurevector)
                    y.append(directory)

    return X, y

TRAINING_DATA = 'training_data/'

#Get training data
X, y = get_training_data(TRAINING_DATA)

#Split training data
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3, random_state=42)

#Train classifier
clf = RandomForestClassifier(n_estimators=200)
clf = clf.fit(X_train, y_train)
s = pickle.dump(clf, open('trained_model.p','w'))

#Classify holdouts
y_pred = clf.predict(X_test)

#Evaluate model
print "Weighted F1 score = %f" %f1_score(y_test, y_pred, pos_label=None, average='weighted')
print "F1 score = "
print f1_score(y_test, y_pred, average=None)

print "Confusion matrix = "
print confusion_matrix(y_test, y_pred)

print "Score = %f" %clf.score(X_test, y_test)
