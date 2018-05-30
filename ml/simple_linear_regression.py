"""
Module for the simple linear regression, the simplest machine to create a model.
"""
import pandas as pd
from sklearn.linear_model import LinearRegression


def train_the_machine(training_values, training_target):
    regressor = LinearRegression()
    regressor.fit(training_values, training_target)
    return regressor


def error(expected, actual):
    df = pd.DataFrame(expected, columns=['Target'])
    df['Predicted'] = pd.DataFrame(actual)
    df['err'] = abs(df['Target'] - df['Predicted'])/df['Target']
    return df, df.describe()['err']['max']


def predict(training_values, training_targets):
    machine = train_the_machine(training_values, training_targets)
    return machine.predict(training_values)



