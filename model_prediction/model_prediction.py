from feature_generation import *

def get_classification(predicted_probabilities, probability_limit):
    """
    Evaluate the predicted probabilities for each class and predict the image type based on a required certainty limit

    :param predicted_probabilities: array with probabilities for each class
    :param probability_limit: threshold required for a classification to be made
    :return: classification
    """

    if predicted_probabilities[0][0] >= probability_limit:
        return 'Plot'
    elif predicted_probabilities[0][1] >= probability_limit:
        return 'Table'
    elif predicted_probabilities[0][2] >= probability_limit:
        return 'Other'
    else:
        return 'Unknown'

def get_prediction(image, trained_model, probability_limit):
    """

    Get the predicted probabilities for each class

    :param image: image to evaluate
    :param model: trained model
    :return: image name, probability for each class and overall classification
    """
    test_feature_vector = get_feature_vector(image)

    predicted_probabilities = trained_model.predict_proba(test_feature_vector)
    classification = get_classification(predicted_probabilities, probability_limit)

    return image, predicted_probabilities[0][0], predicted_probabilities[0][1], predicted_probabilities[0][2], classification