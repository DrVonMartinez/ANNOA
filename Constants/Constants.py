from scipy.stats import norm, uniform, expon, cauchy, laplace, logistic, rayleigh

STATS_SET = [norm, uniform, expon, laplace, logistic, cauchy, rayleigh]
SIZE_SET = [10, 25, 50, 75, 80, 100, 125, 150, 200]
DIMENSION_SET = [1, 2]
SEED = 9
MONTE_CARLO = 10 ** 5
REFERENCE_DICTIONARY = {'Normal': norm, 'Uniform': uniform, 'Exponential': expon, 'Cauchy': cauchy, 'Laplace': laplace, 'Logistic': logistic, 'Rayleigh': rayleigh}
METRIC_SET = ['Loss', 'Mean Absolute Error', 'Accuracy']
PCA_VAL = 8

SHOW = False


def label(byte):
    key = ['B', 'KB', 'MB', 'GB']

    for j in range(1, 4):
        if byte / (10 ** (3 * j)) < 1:
            return byte / (10 ** (3 * (j - 1))), key[j - 1]
    return byte / 10 ** 9, key[-1]
