import argparse
from ocr_prediction import *

parser = argparse.ArgumentParser()
parser.add_argument('images', nargs='*', help='Image files to classify')
args = parser.parse_args()

print 'Image name\tText classification'

for image in args.images:
    text_classification = get_text_classification(image)

    print "%s\t%s" % (image, text_classification)