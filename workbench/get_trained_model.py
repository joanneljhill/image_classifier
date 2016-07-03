from model_evaluation import *
from feature_generation import *
from model_training import *

#Get training data
X, y = get_training_data('training_data/')

#Split training data
X_train, X_test, y_train, y_test = split_data(X, y, 0.3)

#Determine n_estimators to use
#evaluate_n_estimators(X_train, X_test, y_train, y_test)

#Train classifier
train_random_forest(X_train, y_train, 200, 'model1.p')

#Evaluate model
evaluate('model1.p', X_test, y_test)