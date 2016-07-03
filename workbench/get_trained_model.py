from model_evaluation import *
from feature_generation import *
from model_training import *

def get_training_data(training_data_directory):
    """
    Takes labelled folders of images, featurizes them and returns training data that can be used to train models

    :param training_data_directory: directory where the training data is stored
    :return: X and y
    """

    X = []
    y = []

    for directory in os.listdir(training_data_directory):
        if os.path.isdir(training_data_directory + directory + '/'):
            for file in os.listdir(training_data_directory + directory + '/'):
                if not file.startswith('.'):

                    featurevector = get_feature_vector(training_data_directory + directory + '/' + file)

                    X.append(featurevector)
                    y.append(directory)

    return X, y

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