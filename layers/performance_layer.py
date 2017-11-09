"""
Contains functions for setting up the performance evaluation layer for a tensorflow-based RNN.

Copyright (c) 2017 Frank Derry Wanye

Date: 6 November, 2017
"""
import tensorflow as tf
import numpy as np
from copy import deepcopy

from .. import constants

class PerformanceVariables(object):
    """
    Stores and builds the variables needed for calculating performance at the end of an epoch.

    Instance variables:
    - inputs (list): The aggregated inputs from all minibatches
    - labels (list): The aggregated labels from all minibatches
    - sizes (list): The size of each sequence in the labels
    - max_length (int): The length of the longest sequence in the labels
    - current_batch_size (int): The number of sequences in last appended minibatch that are not made of padding
    - input_shape (list/tuple): The shape of the input minibatches (used for padding)
    - label_shape (list/tuple): The shape of the label minibatches (used for padding)
    - input_type (type): The type of data stored in the inputs
    - label_type (type): The type of data stored in the inputs
    - pad_value (label_type): The element of data to be used for padding
    """

    def __init__(self, max_length, shapes, types, pad):
        """
        Creates a PerformanceVariables object.

        Params:
        max_length (int): The maximum length of each sequence in the data
        shapes (list/tuple):
            - input_shape (list/tuple): The shape of the inputs in this data
            - label_shape (list/tuple): The shape of the labels in this data
        types (list/tuple):
            - input_type (type): The type of data stored in the inputs
            - label_type (type): The type of data stored in the labels
        pad (label_type): The element of data to be used for padding
        """
        self.inputs = list()
        self.labels = list()
        self.sizes = list()
        self.max_length = max_length
        self.input_shape = shapes[0]
        self.label_shape = shapes[1]
        self.input_type = types[0]
        self.label_type = types[1]
        self.pad_value = pad
    # End of __init__()

    def add_batch(self, inputs, labels, sizes, beginning, ending):
        """
        Adds a given minibatch to the validation variables data.

        Params:
        inputs (list): The inputs for this minibatch
        labels (list): The labels for this minibatch
        sizes (list): The true size of each sequence for this minibatch
        beginning (boolean): True if this minibatch marks the beginning of a sequence
        ending (boolean): True if this minibatch marks the ending of a sequence
        """
        if beginning is True:
            self.append_batch(inputs, labels, sizes)
        else:
            self.extend_batch(inputs, labels, sizes)
        if ending is True:
            self.pad_batch()
    # End of add_batch()

    def append_batch(self, inputs, labels, sizes):
        """
        Appends a batch that marks the beginning of a sequence to the variables data.

        Creates the following instance variables:
        - current_batch_size (int): The number of sequences in this minibatch that are not made of padding

        Modifies the following instance variables:
        - inputs
        - labels
        - sizes

        Params:
        inputs (list): The inputs for this minibatch
        labels (list): The labels for this minibatch
        sizes (list): The true size of each sequence for this minibatch
        """
        x, y, s = self.copy_batch(inputs, labels, sizes)
        # Do not append padded rows from batches
        self.current_batch_size = len(list(filter(lambda x : x > 0, sizes)))
        self.inputs.append(x[:self.current_batch_size])
        self.labels.append(y[:self.current_batch_size])
        self.sizes.append(s[:self.current_batch_size])
    # End of append_batch()

    def extend_batch(self, inputs, labels, sizes):
        """
        Extends the variables data for the most recently appended sequences with the given minibatch.
        Modifies the following instance variables:
        - inputs
        - labels
        - sizes

        Params:
        inputs (list): The inputs for this minibatch
        labels (list): The labels for this minibatch
        sizes (list): The true size of each sequence for this minibatch
        """
        if len(inputs) != len(labels) or len(inputs) != len(sizes):
            raise ValueError("Members of batch must all have the same first dimension (number of rows) "
                            "inputs=%s | labels=%s | sizes=%s" % (np.shape(inputs), np.shape(labels), np.shape(sizes)))
        x, y, s = self.copy_batch(inputs, labels, sizes)
        for index in range(self.current_batch_size, 0, -1):
            self.inputs[-1][-index].extend(x[-index])
            self.labels[-1][-index].extend(y[-index])
            self.sizes[-1][-index] += s[-index]
    # End of extend_batch()

    def copy_batch(self, inputs, labels, sizes):
        """
        Makes a copy of the given minibatch, so that changes to it do not affect the real dataset values.

        Params:
        inputs (list): The inputs for this minibatch
        labels (list): The labels for this minibatch
        sizes (list): The true size of each sequence for this minibatch

        Return:
        inputs (list): The copy of inputs for this minibatch
        labels (list): The copy of labels for this minibatch
        sizes (list): The copy of true size of each sequence for this minibatch
        """
        x = deepcopy(inputs)
        y = deepcopy(labels)
        s = deepcopy(sizes)
        return x, y, s
    # End of copy_batch()

    def pad_batch(self):
        """
        Pads the variable data for the most recently appended sequences until they are of the same length.
        """
        x_pad = self.batch_padding(self.input_shape, self.input_type)
        y_pad = self.batch_padding(self.label_shape, self.label_type)
        sizes = [0 for s in range(len(y_pad))]
        while len(self.labels[-1][-1]) < self.max_length: # Doesn't matter if using inputs or labels here
            self.extend_batch(x_pad, y_pad, sizes)
    # End of pad_batch()

    def batch_padding(self, shape, pad_type, alt_pad_value=None):
        """
        Creates a full batch made entirely of padding data.

        Params:
        shape (list): The shape of the batch to be created
        pad_type (type): The type of the elements in the minibatch to be created
        alt_pad_value (type): The value the minibatch is to be filled with. Defaults to self.pad_value

        Return:
        padding_batch (list): The padding batch
        """
        if alt_pad_value is None:
            batch = np.full(shape=shape, fill_value=self.pad_value, dtype=pad_type)
        else:
            batch = np.full(shape=shape, fill_value=alt_pad_value, dtype=pad_type)
        return batch.tolist()
    # End of batch_padding()

    def complete(self):
        """
        Normalizes the data by reducing its dimensions to 2 (number of sequences, maximum sequence length).
        Modifies the following instance variables:
        - inputs
        - labels
        - sizes
        """
        self.inputs = self.breakdown(self.inputs)
        self.labels = self.breakdown(self.labels)
        self.sizes = self.breakdown(self.sizes)
        self.inputs = [row[:self.max_length] for row in self.inputs]
        self.labels = [row[:self.max_length] for row in self.labels]
    # End of complete()

    def breakdown(self, batched_values):
        """
        Normalizes a given potion of the data by removing the separation between minibatches of separate sequences.

        Params:
        batched_values (list): The values that have separation between minibatch sequences

        Return:
        unbatched_values (list): The values without the separation between minibatch sequences
        """
        values = list()
        for batch in batched_values:
            values.extend(batch)
        return values
    # End of breakdown()
# End of PerformanceVariables()

def performance_op(logits_series, labels_series, sizes_series):
    loss_op = calculate_loss_op(logits_series, labels_series, sizes_series)
    return loss_op
# End of performance_op()

def calculate_loss_op(logits_series, labels_series, sizes_series):
    """
    Calculates the loss at a given epoch.

    Params:
    logits_series (tf.Tensor): Calculated probabilities for each class for each input after training
    labels_series (tf.Tensor): True labels for each input
    sizes_series (tf.Tensor): The true, un-padded lengths of each row in the minibatch

    Return:
    logits_series
    tf.Tensor: The calculated average loss for this minibatch
    """
    with tf.variable_scope(constants.LOSS_CALC):
        logits_series = tf.unstack(logits_series)
        labels_series = tf.unstack(labels_series)
        sizes_series = tf.unstack(sizes_series)
        total_loss_op = calculate_minibatch_loss(logits_series, labels_series, sizes_series, constants.LOSS_CALC)
    return total_loss_op
# End of calculate_loss()

def calculate_minibatch_loss(logits_series, labels_series, row_lengths_series, scope=constants.BATCH_LOSS_CALC):
    """
    Calculates the loss at a given minibatch.

    Params:
    logits_series (tf.Tensor): Calculated probabilities for each class for each input after training
    labels_series (tf.Tensor): True labels for each input
    row_lengths_series (tf.Tensor): The true, un-padded lengths of each row in the minibatch
    scope (string): The scope under which the operation will appear on the tensorflow graph

    Return:
    tf.Tensor: The calculated average loss for this minibatch
    """
    with tf.variable_scope(scope):
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
        batch_loss_op = loss_sum / num_valid_rows # Can't use reduce_mean because there will be 0s there
    return batch_loss_op
# End of calculate_minibatch_loss()

def calculate_accuracy(logits_series, labels_series, sizes_series):
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
