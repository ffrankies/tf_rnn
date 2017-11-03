"""
Contains functions for setting up the performance evaluation layer for a tensorflow-based RNN.

Copyright (c) 2017 Frank Derry Wanye

Date: 3 November, 2017
"""
import tensorflow as tf
import numpy as np

from .. import constants

class PerformanceVariables(object):
    """
    Stores and builds the variables needed for calculating performance at the end of an epoch
    """

    def __init__(self, max_length, shapes, types, pad):
        """
        Creates a PerformanceVariables object.
        """
        self._inputs = list()
        self._labels = list()
        self._sizes = list()
        self.max_length = max_length
        self.input_shape = shapes[0]
        self.label_shape = shapes[1]
        self.input_type = types[0]
        self.label_type = types[1]
        self.pad_value = pad
    # End of __init__()

    def batch_padding(self, shape, type):
        """
        Creates a full batch made entirely of padding data.
        """
        batch = np.full(shape=shape, fill_value=self.pad_value, dtype=type)
        return batch.tolist()
    # End of batch_padding()

    def add_batch(self, inputs, labels, sizes, beginning, ending):
        """
        Adds a given batch to the validation variables data.
        """
        if beginning is True:
            self._inputs.append(inputs)
            self._labels.append(labels)
            self._sizes.append(sizes)
        else:
            self._inputs[-1].extend(inputs)
            self._labels[-1].extend(labels)
            for index, size in enumerate(sizes):
                self._sizes[-1][index] += size
        if ending is True:
            while len(self._labels[-1]) < self.max_length:
                self._inputs[-1].extend(self.batch_padding(self.input_shape, self.input_type))
                self._labels[-1].extend(self.batch_padding(self.label_shape, self.label_type))
            self._inputs = [row[:self.max_length] for row in self._inputs]
            self._labels = [row[:self.max_length] for row in self._labels]
    # End of add_batch()

    def inputs(self):
        inputs = list()
        for batch in self._inputs:
            inputs.extend(batch)
        return inputs

    def labels(self):
        labels = list()
        for batch in self._labels:
            labels.extend(batch)
        return labels

    def sizes(self):
        sizes = list()
        for batch in self._sizes:
            sizes.extend(batch)
        return sizes
# End of PerformanceVariables()

def calculate_accuracy(labels_series, predictions_series):
    """
    Tensorflow operation that calculates the model's accuracy on a given minibatch.

    Params:
    labels_series (tf.Tensor): True labels for each input
    predictions_series (tf.Tensor): The predictions made by the RNN for each input

    Return:
    tf.Tensor: The average accuracy for each row in the minibatch
    """
    with tf.variable_scope(constants.ACCURACY):
        accuracy = []
        for predictions, labels in zip(predictions_series, labels_series):
            labels = tf.to_int64(labels, "CastLabelsToInt")
            predictions = tf.argmax(predictions, axis=1)
            accuracy.append(tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32)))
    return accuracy
# End of calculate_accuracy()

def calculate_loss(logits_series, labels_series, row_lengths_series):
    """
    Calculates the loss at a given minibatch.

    Params:
    logits_series (tf.Tensor): Calculated probabilities for each class for each input after training
    labels_series (tf.Tensor): True labels for each input
    row_lengths_series (tf.Tensor): The true, un-padded lengths of each row in the minibatch

    Return:
    tf.Tensor: The calculated average loss for this minibatch
    """
    with tf.variable_scope(constants.LOSS_CALC):
        logits_series = tf.placeholder(dtype=tf.float32, shape=np.shape(logits_series), name="logits_placeholder")
        labels_series = tf.placeholder(dtype=tf.int32, shape=np.shape(labels_series), name="labels_placeholder")
        sizes_series = tf.placeholder(dtype=tf.int32, shape=np.shape(row_lengths_series), name="sizes_series")

        loss_sum = 0.0
        num_valid_rows = 0.0
        for logits, labels, row_length in zip(logits_series, labels_series, row_lengths_series):
            # row_length = tf.to_int32(row_length, name="CastRowLengthToInt")
            ans = tf.greater(row_length, 0)
            num_valid_rows = tf.cond(ans, lambda: num_valid_rows + 1, lambda: num_valid_rows + 0)
            logits = logits[:row_length, :]
            labels = tf.to_int32(labels[:row_length], "CastLabelsToInt")
            row_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            mean_loss = tf.cond(ans, lambda: tf.reduce_mean(row_losses[:row_length]), lambda: 0.0)
            loss_sum += mean_loss
        total_loss_op = loss_sum / num_valid_rows # Can't use reduce_mean because there will be 0s there
    return total_loss_op
# End of calculate_loss()

def append_minibatch(all_inputs, all_outputs, new_inputs, new_outputs):
    """
    Builds a large matrix from the inputs and outputs.
    """
