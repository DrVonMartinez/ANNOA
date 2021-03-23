from keras.metrics import mae, Recall, Precision, TruePositives, TrueNegatives, FalsePositives, FalseNegatives

OPTIMIZER_SET = ['adadelta', 'adam', 'ftrl', 'nadam', 'sgd']
EXPANDED_MODEL_METRICS = [mae, 'accuracy', Precision(), Recall(), TruePositives(), TrueNegatives(), FalsePositives(), FalseNegatives()]
