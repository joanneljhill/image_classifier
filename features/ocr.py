from pytesseract import image_to_string
from PIL import Image

def get_text(image):
    """
    Extract text from a given image

    :param image: path to image from which to extract text
    :return: text extracted from image
    """

    img = Image.open(image)
    return image_to_string(img).replace('\n',' ').lower()

def get_text_classification(image):
    """
    Classify image based on text in the image

    :param image: path to image to classify
    :return: classification
    """

    text = get_text(image)

    if 'table ' in text:
        return 'Table'
    elif 'micrograph' in text:
        return 'Other'
    elif 'plot' in text or 'graph' in text:
        return 'Plot'
    else:
        return "Unknown"