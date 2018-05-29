"""
Data preprocessing module
"""
from os import sep
import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


FOLDER = 'Data_Preprocessing'
DATA_FILE = 'Data.csv'


def import_data(filename=f'.{sep}{FOLDER}{sep}{DATA_FILE}'):
    return pd.read_csv(filename)


def features_and_dependent_vars(dataset):
    """Returns the features matrix and dependet variable array
       from the given dataset
    """
    return dataset.iloc[:, :-1].values, dataset.iloc[:, -1:].values


def fetch_slice(matrix, rows, cols):
    return matrix[rows, cols]


def cleanup_data(data):
    matrix, dep_array = features_and_dependent_vars(data)
    slice_rows = slice(None, None)
    slice_cols = [1, 2]
    data_to_change = fetch_slice(matrix, slice_rows, slice_cols)
    return update_slice(matrix, data_to_change, slice_rows, slice_cols), dep_array


def update_slice(matrix, data_to_change, rows, cols):
    imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    matrix[rows, cols] = imputer.fit(data_to_change).transform(data_to_change)
    return matrix


def encode_feature(matrix, rows=slice(None, None), col=0):
    encoder = LabelEncoder()
    matrix[rows, col] = encoder.fit_transform(matrix[rows, col])
    return matrix


def create_dummy_variables(matrix, columns=[0]):
    spread_encoder = OneHotEncoder(categorical_features=columns)
    return spread_encoder.fit_transform(matrix).toarray()


def encode_data(data):
    matrix, dep_vars = cleanup_data(data)
    feature_matrix_with_dummies = create_dummy_variables(encode_feature(matrix))
    dependent_var_array = encode_feature(dep_vars)
    return matrix, feature_matrix_with_dummies, dependent_var_array


def training_set(matrix, dependent_array, test_size=0.2, random_state=0):
    """Creates the training and test sets from the given data"""
    return train_test_split(matrix, dependent_array, test_size=test_size, random_state=random_state)


def feature_scaling(train_set, test_set):
    scaler = StandardScaler()
    return scaler.fit_transform(train_set), scaler.transform(test_set)


def run(): 
    matrix, feature_matrix_with_dummies, dependent_var_array = encode_data(import_data())
    train_set, test_set, dep_train_set, dep_test_set = training_set(feature_matrix_with_dummies, dependent_var_array)
    scaled_train_set, scaled_test_set = feature_scaling(train_set, test_set)
    return scaled_train_set, scaled_test_set, dep_train_set, dep_test_set
