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

    def max_error():
        return dataframe.describe()['err']['max']

    dataframe = pd.DataFrame(expected, columns=['Target'])
    dataframe['Predicted'] = pd.DataFrame(actual)
    dataframe['err'] = abs(dataframe['Target'] - dataframe['Predicted'])/dataframe['Target']
    return dataframe, max_error()


def predict(machine, test_values):
    return machine.predict(test_values)


def chart_plot(plt, training_set, test_set, dataset_type):

    def labels_and_title():
        plt.title(f'Salary vs Experience ({dataset_type} Set)')
        plt.xlabel('Years of Experience')
        plt.ylabel('Salary')

    plt.scatter(*plot_set, color='red')
    plt.plot(training_set[0], predict(train_the_machine(*training_set), training_set[0]))
    labels_and_title()
    plt.show()
