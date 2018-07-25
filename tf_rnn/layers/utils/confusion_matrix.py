"""The confusion matrix class.

@since 0.6.1
"""

from collections import namedtuple
from copy import deepcopy

import numpy as np

# The following imports are only used for type hinting
from ...logger import Logger


PerformanceMetrics = namedtuple('PerformanceMetrics', ['accuracy', 'precision', 'recall', 'f1_score'])


class ConfusionMatrix(object):
    """An updatable confusion matrix.

    Instance Variables:
    - logger (logging.Logger): For logging purposes
    - matrix (dict): The confusion matrix, as a dictionary
    - row_labels (set): The valid labels for the rows
    - col_labels (set): The valid labels for the columns
    """

    def __init__(self, logger: Logger):
        """Creates a ConfusionMatrix object.
        """
        self.logger = logger
        # logger.debug('Creating a confusion matrix')
        self.matrix = dict()
        self.row_labels = set()
        self.col_labels = set()
    # End of __init__()

    def update(self, predictions: list, labels: list, sequence_lengths: list):
        """Updates the confusion matrix using the given predictions and their correct labels for one batch.

        Params:
        - predictions (list): The predictions made by the neural network for the given batch
        - labels (list): The correct predictions for the given batch
        - sequence_lengths (list): The valid sequence lengths for every row in the given batch
        """
        # self.logger.debug('Updating confusion matrix')
        for row_index, row in enumerate(predictions):
            row_slice = row[:sequence_lengths[row_index]]
            for column_index, prediction in enumerate(row_slice):
                label = labels[row_index][column_index]
                self.insert_prediction(prediction, label)
    # End of update()

    def insert_prediction(self, prediction: int, label: int):
        """Inserts a prediction into the confusion matrix.

        Params:
        - prediction (int): The predictions to insert into the matrix
        - label (int): The label for the prediction to be inserted
        """
        if label not in self.matrix.keys():
            self.matrix[label] = dict()
        if prediction not in self.matrix[label].keys():
            self.matrix[label][prediction] = 1
        else:
            self.matrix[label][prediction] += 1
        self.row_labels.add(label)
        self.col_labels.add(prediction)
    # End of insert_prediction()

    def to_array(self) -> list:
        """Converts the confusion matrix dictionary to a 2d array.

        Returns:
        - confusion_matrix (list): A 2d array representation of the confusion matrix
        """
        confusion_matrix = list()
        labels = self.all_labels()
        for row_label in labels:
            if row_label in self.matrix:
                row_dict = self.matrix[row_label]
                row = [row_dict[col_label] if col_label in row_dict.keys() else 0 for col_label in labels]
            else:
                row = [0 for label in labels]
            confusion_matrix.append(row)
        return confusion_matrix
    # End of to_array()

    def all_labels(self) -> set:
        """Returns a set of all labels in the confusion matrix.

        Returns:
        - labels (set): The set of all labels in the confusion matrix
        """
        labels = set(self.row_labels)
        labels.update(self.col_labels)
        return labels
    # End of all_labels()

    def to_normalized_array(self) -> np.array:
        """Converts the confusion matrix dictionary to a normalized 2d array.
        Normalization happens across rows, and all resulting values are between 0 and 1.

        Returns:
        - normalized_confusion_matrix (list): A 2d array representation of the normalized confusion matrix
        """
        confusion_matrix = self.to_array()
        sums = np.sum(confusion_matrix, axis=1)  # Sum matrix along row
        normalized_confusion_matrix = np.divide(confusion_matrix, sums[:, np.newaxis])  # Divide each value by row total
        return normalized_confusion_matrix
    # End of to_normalized_array()

    def copy(self) -> object:
        """Creates a copy of this confusion matrix object.

        Returns:
        - confusion_matrix_copy (ConfusionMatrix): A copy of this confusion matrix
        """
        copy = ConfusionMatrix(self.logger)
        copy.row_labels = deepcopy(self.row_labels)
        copy.col_labels = deepcopy(self.col_labels)
        copy.matrix = deepcopy(self.matrix)
        return copy
    # End of copy()

    def is_empty(self) -> bool:
        """Specifies whether or not the confusion matrix is empty.

        Returns:
        - is_empty (bool): True if the confusion matrix is empty, False otherwise
        """
        return len(self.matrix) == 0
    # End of is_empty()

    def performance_metrics(self) -> PerformanceMetrics:
        """Calculates the performance metrics for this confusion matrix.

        Returns:
        - performance_metrics (PerformanceMetrics): The performance matrix for this confusion matrix
        """
        matrix = self.to_array()
        matrix = np.array(matrix)
        identity = np.identity(len(self.all_labels))
        masked_matrix = matrix * identity
        accuracy = np.sum(masked_matrix) / np.sum(matrix)
        with np.errstate(divide='ignore'):
            # recall = true positives / true positives + false negatives
            recall = masked_matrix.sum(axis=0) / matrix.sum(axis=0)
            # precision = true positives / true positives + false positives
            precision = masked_matrix.sum(axis=1) / matrix.sum(axis=1)
        recall = np.nan_to_num(recall).mean()
        precision = np.nan_to_num(recall).mean()
        f1_score = (2 * recall * precision) / (recall + precision)
        return PerformanceMetrics(accuracy, precision, recall, f1_score)
    # End of performance_matrix()
# End of ConfusionMatrix()
