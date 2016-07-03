import argparse
import pickle
from model_prediction import *

#get image files to classify
parser = argparse.ArgumentParser()
parser.add_argument('images', nargs='*', help='Image files to classify')
args = parser.parse_args()

#print headings
print 'Image name\t0 - Plot\t1 - Table\t2 - Other\tClassification'

#load trained model
clf = pickle.load(open('model1.p'))

#classify each image
for image in args.images:
    name, plot, table, other, classification = get_prediction(image, clf, 0.6)
    print "%s\t%f\t%f\t%f\t%s" % (name, plot, table, other, classification)