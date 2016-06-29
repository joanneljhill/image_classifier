import sys
import pickle
from PIL import Image
import numpy as np
import argparse
from pytesseract import image_to_string

def get_overall_classification(ml_classification, text_classification):
    if ml_classification == text_classification:
        return ml_classification
    elif ml_classification == 'Unknown' and text_classification != 'Unknown':
        return text_classification
    elif text_classification == 'Unknown' and ml_classification != 'Unknown':
        return ml_classification
    else:
        return "Unknown"

def get_text(image):
    img = Image.open(image)
    return image_to_string(img).replace('\n',' ').lower()

def get_text_classification(text):
    if 'table ' in text:
        return 'Table'
    elif 'micrograph' in text:
        return 'Other'
    elif 'plot' in text or 'graph' in text:
        return 'Plot'
    else:
        return "Unknown"

def get_featurevector(image):
    img = Image.open(image).convert('L')
    new_img = img.resize((60,30))

    return np.array(new_img).flatten()

def get_ml_classification(prediction, probability):
    if prediction[0][0] >= probability:
        return 'Plot'
    elif prediction[0][1] >= probability:
        return 'Table'
    elif prediction[0][2] >= probability:
        return 'Other'
    else:
        return 'Unknown'

def get_ml_prediction(image):
    test_featurevector = get_featurevector(image)

    prediction = clf.predict_proba(test_featurevector)
    classification = get_ml_classification(prediction, 0.75)

    return image, prediction[0][0], prediction[0][1], prediction[0][2], classification

parser = argparse.ArgumentParser()
parser.add_argument('images', nargs='*', help='Image files to classify')
args = parser.parse_args()

clf = pickle.load(open('model1.p'))

print 'Image name\t0 - Plot\t1 - Table\t2 - Other\tML classification\tText classification\tOverall classification'

for image in args.images:
    name, plot, table, other, ml_classification = get_ml_prediction(image)
    text_classification = get_text_classification(get_text(image))
    overall_classification = get_overall_classification(ml_classification, text_classification)

    print "%s\t%f\t%f\t%f\t%s\t\t\t%s\t\t\t%s" % (name, plot, table, other, ml_classification, text_classification, overall_classification)