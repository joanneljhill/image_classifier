from image_classifier import helpers, training, ocr, image, prediction

import argparse, pickle

# train model
helpers.get_trained_model('/Users/joannehill/ml/image_classifier/training_data/', 'model1.p')

# get image files to classify
parser = argparse.ArgumentParser()
parser.add_argument('images', nargs='*', help='Image files to classify')
args = parser.parse_args()

# load trained model
clf = pickle.load(open('model1.p'))

# classify test images
for image in args.images:
    helpers.get_text_classification(image)
    helpers.get_image_classification(image, clf)
