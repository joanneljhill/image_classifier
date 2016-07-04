import features
import training
import evaluation
import os

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

                    featurevector = features.get_feature_vector(training_data_directory + directory + '/' + file)

                    X.append(featurevector)
                    y.append(directory)

    return X, y

def get_trained_model(training_data_directory, model_name):

    #Get training data
    X, y = get_training_data(training_data_directory)

    #Split training data
    X_train, X_test, y_train, y_test = training.training.split_data(X, y, 0.3)

    #Determine n_estimators to use
    #evaluate_n_estimators(X_train, X_test, y_train, y_test)

    #Train classifier
    training.train_random_forest(X_train, y_train, 200, model_name)

    evaluation.evaluate(model_name, X_test, y_test)



