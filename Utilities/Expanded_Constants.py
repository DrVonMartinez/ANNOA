from scipy.stats import norm, uniform, expon, cauchy, laplace, logistic, rayleigh

REFERENCE_DICTIONARY = {'Normal': norm, 'Uniform': uniform, 'Exponential': expon, 'Cauchy': cauchy, 'Laplace': laplace, 'Logistic': logistic, 'Rayleigh': rayleigh}
REFERENCE_LIST = ['Normal', 'Uniform', 'Exponential', 'Cauchy', 'Laplace', 'Logistic', 'Rayleigh']
NUM_HIDDEN_LAYERS = range(0, 3)
HIDDEN_NEURONS = 250
OPTIMIZER_SET = ['adadelta', 'adam', 'ftrl', 'nadam', 'sgd']
NUM_EPOCHS = 50
EXPANDED_METRIC_SET = ['Loss', 'Mean Absolute Error', 'Accuracy', 'Precision', 'Recall', 'True Positives', 'True Negatives', 'False Positives', 'False Negatives']
EXPANDED_HISTORY_KEYS = ['loss', 'mean_absolute_error', 'accuracy', 'precision', 'recall', 'true_positives', 'true_negatives', 'false_positives', 'false_negatives']
