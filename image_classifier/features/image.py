from PIL import Image
import numpy as np

def get_feature_vector(image):
    """
    Converts an image to grayscale then resizes the image to 60 x 30

    :param image: path to an image file that needs to be featurized
    :return: flattened numpy array
    """

    img = Image.open(image).convert('L')
    new_img = img.resize((60,30))

    return np.array(new_img).flatten()