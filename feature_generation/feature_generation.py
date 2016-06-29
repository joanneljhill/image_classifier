import os
from PIL import Image
import numpy as np

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

                    featurevector = get_featurevector(training_data_directory + directory + '/' + file)

                    X.append(featurevector)
                    y.append(directory)

    return X, y

def get_feature_vector(image):
    """
    Converts an image to grayscale then resizes the image to 60 x 30

    :param image: path to an image file that needs to be featurized
    :return: flattened numpy array
    """

    img = Image.open(image).convert('L')
    new_img = img.resize((60,30))

    return np.array(new_img).flatten()