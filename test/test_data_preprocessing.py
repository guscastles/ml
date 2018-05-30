"""
Test module for data preprocessing
"""
import pytest
import numpy as np
from ml.data_preprocessing import features_and_dependent_vars, cleanup_data, import_data, \
                                  create_dummy_variables, encode_feature, training_set, \
                                  encode_data, feature_scaling, run


def test_features_matrix():
    matrix, dep_vars = features_and_dependent_vars(import_data())
    assert matrix[0][0] == 'France'
    assert dep_vars[0][0] == 'No'


def test_import_data():
    dataset = import_data() 
    assert dataset is not None


@pytest.mark.data
def test_cleanup_data():
    matrix, _ = cleanup_data(import_data())
    assert matrix[4, 2] == 63777.77777777778


@pytest.mark.data
def test_categorical_data():
    matrix = encode_feature(cleanup_data(import_data())[0], slice(None, None), 0)
    assert set(matrix[:, 0]) == {0, 1, 2}
    assert create_dummy_variables(matrix).shape == (10, 5)


def test_encode_data():
    original_data, data_with_dummies, dependent_vars = encode_data(import_data())
    assert original_data[6, 2] is not np.nan
    assert data_with_dummies.shape == (10, 5)
    assert dependent_vars[0] == 0


def test_training_set():
    matrix_train, matrix_test, dependent_train, dependent_test = training_set(*features_and_dependent_vars(import_data()))
    assert matrix_train.shape == (8, 3)
    assert matrix_test.shape == (2, 3)
    assert dependent_train.shape == (8, 1)
    assert dependent_test.shape == (2, 1)


@pytest.mark.wip
def test_feature_scaling():
    matrix, dep_vars = cleanup_data(import_data())
    matrix_train, matrix_test, dependent_train, dependent_test = training_set(encode_feature(matrix), 
                                                                              encode_feature(dep_vars))
    scaled_matrix_train, scaled_matrix_test = feature_scaling(matrix_train, matrix_test) 
    assert scaled_matrix_train.shape == matrix_train.shape


def test_run():
    train_set, test_set, dep_train_set, dep_test_set = run()
    assert train_set.shape == (8, 5)
