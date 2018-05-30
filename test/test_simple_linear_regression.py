"""
Test module for the simple linear regression, the simplest machine to create a model.
"""
import pytest
import pandas as pd
from ml import data_preprocessing as dp
from ml.simple_linear_regression import predict, error, train_the_machine
from sklearn.linear_model import LinearRegression


DATA_FILE = 'Machine Learning A-Z Template Folder/Part 2 - Regression/Section 4 - Simple Linear Regression/Simple_Linear_Regression/Simple_Linear_Regression/Salary_Data.csv'


@pytest.mark.linreg
def test_read_data():
    data = dp.import_data(DATA_FILE)
    assert data.iloc[0, 0] == 1.1
    matrix, depend = dp.features_and_dependent_vars(data)
    assert matrix[0][0] == 1.1 and depend[0][0] == 39343


@pytest.mark.linreg
def test_training_and_test_sets():
    train_x, test_x, train_y, test_y = dp.training_set(*dp.features_and_dependent_vars(dp.import_data(DATA_FILE)), test_size=1/3)
    assert train_x[0][0] == 2.9
    assert test_x[0][0] == 1.5
    assert train_y[0][0] == 56642
    assert test_y[0][0] == 37731


@pytest.mark.linreg
def test_train_the_machine():
    train_x, _, train_y, _ = dp.training_set(*dp.features_and_dependent_vars(dp.import_data(DATA_FILE)), test_size=1/3)
    machine = train_the_machine(train_x, train_y)
    assert isinstance(machine, LinearRegression)


@pytest.mark.linreg
def test_predict():
    train_x, test_x, train_y, test_y = dp.training_set(*dp.features_and_dependent_vars(dp.import_data(DATA_FILE)), test_size=1/3)
    machine = train_the_machine(train_x, train_y)
    predicted = predict(machine, train_x)
    data, max_error = error(train_y, predicted)
    assert max_error < 0.16
