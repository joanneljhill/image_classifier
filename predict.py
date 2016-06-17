import sys
import pickle
from PIL import Image
import numpy as np
import argparse

def get_featurevector(image):
    img = Image.open(image).convert('L')
    new_img = img.resize((60,30))

    return np.array(new_img).flatten()

def get_classification(prediction, probability):
    if prediction[0][0] >= probability:
        return 'Plot'
    elif prediction[0][1] >= probability:
        return 'Table'
    elif prediction[0][2] >= probability:
        return 'Other'
    else:
        return 'Unknown'

def get_prediction(image):
    test_featurevector = get_featurevector(image)

    prediction = clf.predict_proba(test_featurevector)
    classification = get_classification(prediction, 0.5)

    return image, prediction[0][0], prediction[0][1], prediction[0][2], classification

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('images', nargs='*', help='Image files to classify')
    args = parser.parse_args()

    clf = pickle.load(open('trained_model.p'))

    print 'Image name\t0 - Plot\t1 - Table\t2 - Other\tPrediction'

    for image in args.images:
        name, plot, table, other, prediction = get_prediction(image)
        print "%s\t%f\t%f\t%f\t%s" % (name, plot, table, other, prediction)