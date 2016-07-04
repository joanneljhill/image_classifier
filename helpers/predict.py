import features
import prediction

def get_text_classification(image):
    print 'Image name\tText classification'

    text_classification = features.get_text_classification(image)
    print "%s\t%s" % (image, text_classification)

def get_image_classification(image, model):
    #print headings
    print 'Image name\t0 - Plot\t1 - Table\t2 - Other\tClassification'

    #classify each image
    name, plot, table, other, classification = prediction.get_prediction(image, model, 0.6)
    print "%s\t%f\t%f\t%f\t%s" % (name, plot, table, other, classification)