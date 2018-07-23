"""Test for the confusion_matrix module.
"""
import pytest
import logging
import numpy as np

from tf_rnn.layers.performance_layer import *
from ...test_data import *

LOGGER = logging.getLogger('TEST')

MAX_LENGTH = 8

COMBINED_SCRAMBLED_BATCHES_4 = [
    [0, 1, 2, 3, 4, 5, 6, 7],
    [0, 1, 2, PAD, PAD, PAD, PAD, PAD],
    [0, 1, PAD, PAD, PAD, PAD, PAD, PAD],
    [0, 1, 2, 3, 4, PAD, PAD, PAD],
    [0, 1, 2, 3, PAD, PAD, PAD, PAD],
    [0, PAD, PAD, PAD, PAD, PAD, PAD, PAD],
    [0, 1, 2, 3, 4, 5, 6, PAD],
    [0, 1, 2, 3, 4, 5, PAD, PAD]
]

SIZES_COMBINED_SCRAMBLED_BATCHES_4 = [8, 3, 2, 5, 4, 1, 7, 6]

PREDICTIONS = COMBINED_SCRAMBLED_BATCHES_4

LABELS = COMBINED_SCRAMBLED_BATCHES_4 = [
    [0, 1, 2, 3, 5, 5, 1, 7],
    [0, 1, 2, PAD, PAD, PAD, PAD, PAD],
    [0, 1, PAD, PAD, PAD, PAD, PAD, PAD],
    [0, 1, 2, 3, 5, PAD, PAD, PAD],
    [1, 2, 3, 4, PAD, PAD, PAD, PAD],
    [1, PAD, PAD, PAD, PAD, PAD, PAD, PAD],
    [1, 2, 3, 4, 5, 5, 6, PAD],
    [1, 2, 3, 4, 5, 5, PAD, PAD]
]

SIZES = SIZES_COMBINED_SCRAMBLED_BATCHES_4

#
# Accumulator tests
#

def make_data_list(batch, size, loss, accuracy, t_accuracy):
    """Creates the data for updating an accumulator.
    """
    max_size = max(size)
    t_size = [0] * max_size
    for i in size:
        for index, j in enumerate(range(i)):
            t_size[index] += 1
    print('t_size = ', t_size)
    data = [
        loss,  # loss
        accuracy,  # accuracy
        sum(size),  # number of elements in batch
        t_accuracy,  # timestep accuracies
        t_size,  # timestep lengths
        batch,  # predictions
        batch,  # labels
        size  # length of each sequence
    ]
    return data
# End of make_data_list()
 

class TestInsertPredictionIntoConfusionMatrix():
    def test_should_correctly_add_new_prediction(self):
        cm = ConfusionMatrix(LOGGER)
        cm.insert_prediction(0, 1)
        assert 1 in cm.row_labels
        assert 0 in cm.col_labels
        assert len(cm.matrix.keys()) == 1
        assert len(cm.matrix[1].keys()) == 1
        assert cm.matrix[1][0] == 1

    def test_should_correctly_update_old_prediction(self):
        cm = ConfusionMatrix(LOGGER)
        cm.insert_prediction(0, 1)
        cm.insert_prediction(0, 1)
        assert 1 in cm.row_labels
        assert 0 in cm.col_labels
        assert len(cm.matrix.keys()) == 1
        assert len(cm.matrix[1].keys()) == 1
        assert cm.matrix[1][0] == 2
        
class TestUpdateConfusionMatrix():
    def test_should_do_nothing_if_data_is_empty(self):
        predictions = []
        labels = []
        sizes = []
        cm = ConfusionMatrix(LOGGER)
        cm.update(predictions, labels, sizes)
        assert len(cm.matrix.keys()) == 0

    def test_should_correctly_update_with_batch_data(self):
        cm = ConfusionMatrix(LOGGER)
        cm.update(PREDICTIONS, LABELS, SIZES)
        assert cm.row_labels == cm.col_labels
        assert PAD not in cm.row_labels and PAD not in cm.col_labels
        assert list(cm.matrix[0].keys()) == [0]
        assert list(cm.matrix[1].keys())== [0, 1, 6]
        assert list(cm.matrix[2].keys())== [1, 2]
        assert list(cm.matrix[3].keys())== [2, 3]
        assert list(cm.matrix[4].keys())== [3]
        assert list(cm.matrix[5].keys())== [4, 5]
        assert list(cm.matrix[6].keys())== [6]
        assert list(cm.matrix[7].keys())== [7]
        assert cm.matrix[0][0] == 4
        assert cm.matrix[1][0] == 4 and cm.matrix[1][1] == 4 and cm.matrix[1][6] == 1
        assert cm.matrix[2][1] == 3 and cm.matrix[2][2] == 3
        assert cm.matrix[3][2] == 3 and cm.matrix[3][3] == 2
        assert cm.matrix[4][3] == 3
        assert cm.matrix[5][4] == 4 and cm.matrix[5][5] == 3
        assert cm.matrix[6][6] == 1
        assert cm.matrix[7][7] == 1

class TestConfusionMatrixToArray():
    def test_should_return_empty_array_when_matrix_is_empty(self):
        cm = ConfusionMatrix(LOGGER)
        matrix = cm.to_array()
        assert matrix == []

    def test_should_correctly_convert_matrix_with_data_to_array(self):
        cm = ConfusionMatrix(LOGGER)
        cm.update(PREDICTIONS, LABELS, SIZES)
        matrix = cm.to_array()
        assert matrix == [
            [4, 0, 0, 0, 0, 0, 0, 0],
            [4, 4, 0, 0, 0, 0, 1, 0],
            [0, 3, 3, 0, 0, 0, 0, 0],
            [0, 0, 3, 2, 0, 0, 0, 0],
            [0, 0, 0, 3, 0, 0, 0, 0],
            [0, 0, 0, 0, 4, 3, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ]
        assert cm.matrix[0][0] == 4
        assert cm.matrix[1][0] == 4 and cm.matrix[1][1] == 4 and cm.matrix[1][6] == 1
        assert cm.matrix[2][1] == 3 and cm.matrix[2][2] == 3
        assert cm.matrix[3][2] == 3 and cm.matrix[3][3] == 2
        assert cm.matrix[4][3] == 3
        assert cm.matrix[5][4] == 4 and cm.matrix[5][5] == 3
        assert cm.matrix[6][6] == 1
        assert cm.matrix[7][7] == 1
