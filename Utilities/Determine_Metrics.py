import numpy as np
import pandas as pd


def false_positive_rate(df: pd.DataFrame):
    fp = df['False Positives']
    tn = df['True Negatives']
    df['False Positive Rate'] = fp / (tn + fp)
    return df


def false_omission_rate(df: pd.DataFrame):
    fn = df['False Negatives']
    tn = df['True Negatives']
    df['False Omission Rate'] = fn / (fn + tn)
    return df


def specificity(df: pd.DataFrame):
    if 'False Positive Rate' in df.columns:
        df['Specificity'] = 1 - df['False Positive Rate']
    else:
        fp = df['False Positives']
        tn = df['True Negatives']
        df['Specificity'] = 1 - fp / (tn + fp)
    return df


def prevalence_threshold(df: pd.DataFrame):
    tpr = df['Recall']
    if 'Specificity' in df.columns:
        tnr = df['Specificity']
    else:
        fp = df['False Positives']
        tn = df['True Negatives']
        tnr = 1 - fp / (tn + fp)
    df['Prevalence Threshold'] = (np.sqrt(tpr * (1 - tnr)) + tnr - 1) / (tpr + tnr - 1)
    return df


def f1_score(df: pd.DataFrame):
    tp = df['True Positives']
    fp = df['False Positives']
    fn = df['False Negatives']
    df['F1 score'] = (2 * tp) / ((2 * tp) + fp + fn)
    return df

