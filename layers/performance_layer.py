"""
Contains functions for setting up the performance evaluation layer for a tensorflow-based RNN.

Copyright (c) 2017 Frank Derry Wanye

Date: 10 November, 2017
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

def performance_op(logits_series, labels_series, sizes_series, max_length):
    """
    Calculates the loss at a given epoch.

    Params:
    logits_series (tf.Tensor): Calculated probabilities for each class for each input after training
    labels_series (tf.Tensor): True labels for each input
    sizes_series (tf.Tensor): The true, un-padded lengths of each row in the minibatch

    Return:
    average_loss (tf.Tensor): The calculated average loss for the given logits
    average_accuracy (tf.Tensor): The calculated average accuracy for the given logits
    """
    logits_h, labels_h, sizes_h = unstack_variables(logits_series, labels_series, sizes_series, True)
    loss_op = calculate_minibatch_loss(logits_h, labels_h, sizes_h, constants.LOSS_CALC)
    masked_predictions, timestep_lengths = predict_and_mask(logits_series, labels_series, sizes_series, max_length)
    accuracy_op = overall_accuracy(masked_predictions, sizes_series)
    timestep_accuracies_op = timestep_accuracy(masked_predictions, timestep_lengths)
    return loss_op, accuracy_op, timestep_accuracies_op
# End of performance_op()

def unstack_variables(logits_series, labels_series, sizes_series, horizontal=True):
    """
    Unstacks the given logits, labels and sizes along the given orientation.

    Params:
    logits_series (tf.placeholder): The logits to be unstacked
    labels_series (tf.placeholder): The labels to be unstacked
    sizes_series (tf.placeholder): The row lengths to be unstacked
    horizontal (boolean): If true, unstack along axis 0, otherwise along axis 1
    """
    if horizontal is True:
        axis = 0
        scope = "horizontal_unstack"
    else:
        axis = 1
        scope = "vertical_unstack"
    with tf.variable_scope(scope):
        unstacked_logits = tf.unstack(logits_series, axis=axis, name="unstack_logits")
        unstacked_labels = tf.unstack(labels_series, axis=axis, name="unstack_labels")
        unstacked_sizes = tf.unstack(sizes_series, axis=axis, name="unstack_labels")
    return unstacked_logits, unstacked_labels, unstacked_sizes
# End of unstack_variables()

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

def predict_and_mask(logits_series, labels_series, sizes_series, max_row_length):
    """
    Finds the correct predictions made across the given logits, and applies a mask so that it only contains
    valid predictions.

    Params:
    logits_series (tf.Tensor): Calculated probabilities for each class for each input
    labels_series (tf.Tensor): True labels for each input
    sizes_series (tf.Tensor): The true, un-padded lengths of each row in the minibatch
    max_row_length (int): The maximum row length

    Return:
    masked_predictions (tf.Tensor): The correct predictions, after the mask has been applied to them
    timestep_lengths (tf.Tensor): The number of valid predictions at each timestep
    """
    with tf.variable_scope(constants.PREDICTIONS_MASK):
        mask, timestep_lengths = row_length_mask(sizes_series, max_row_length)
        predictions = tf.nn.softmax(logits_series, dim=-1, name="logits_softmax")
        predictions = tf.argmax(predictions, axis=-1, name="logits_argmax", output_type=tf.int32)
        correct_predictions = tf.equal(predictions, labels_series)
        correct_predictions = tf.cast(correct_predictions, tf.float32)
        correct_predictions_masked = tf.multiply(correct_predictions, mask)
    return correct_predictions_masked, timestep_lengths
# End of predict_and_mask()

def row_length_mask(sizes_series, max_row_length):
    """
    Constructs a mask out of the row lengths series.

    Params:
    sizes_series (tf.Tensor): The length of each sequence (row) in the data
    max_row_length (int): The maximum length of sequences in the data

    Return:
    mask (tf.Tensor): A mask containing 1s where the logits are valid, 0 where they are not
    timestep_lengths (tf.Tensor): The number of valid logits at each timestep in the data
    """
    mask = tf.sequence_mask(sizes_series, maxlen=max_row_length, dtype=tf.float32, name="row_length_mask")
    timestep_lengths = tf.reduce_sum(mask, axis=0, name="timestep_lengths")
    return mask, timestep_lengths
# End of row_length_mask()

def overall_accuracy(masked_predictions, sizes_series):
    """
    Tensorflow operation that calculates the model's accuracy on a given minibatch.

    Params:
    labels_series (tf.Tensor): True labels for each input
    predictions_series (tf.Tensor): The predictions made by the RNN for each input

    Return:
    tf.Tensor: The average accuracy for each row in the minibatch
    """
    with tf.variable_scope(constants.ACCURACY):
        row_sums = tf.reduce_sum(masked_predictions, axis=1, name="correct_predictions_per_sequence")
        sizes_series = tf.cast(sizes_series, tf.float32, name="cast_row_lengths_to_float32")
        row_accuracies = tf.divide(row_sums, sizes_series, name="row_accuracies")
        row_accuracies = tf.where(tf.is_nan(row_accuracies), sizes_series, row_accuracies)
        average_accuracy = tf.reduce_mean(row_accuracies, name="average_accuracy")
        # accuracy_sum = 0.0
        # for logits, labels, row_length in zip(logits_series, labels_series, sizes_series):
        #     logits = logits[:row_length, :]
        #     labels = labels[:row_length]
        #     predictions = tf.nn.softmax(logits, dim=-1, name="logits_softmax")
        #     predictions = tf.argmax(predictions, axis=-1, name="logits_argmax")
        #     predictions = tf.to_int32(predictions, "CastPredictionsToInt32")
        #     correct_predictions = tf.equal(predictions, labels)
        #     correct_predictions = tf.cast(correct_predictions, tf.float32)
        #     row_accuracy = tf.reduce_mean(correct_predictions)
        #     accuracy_sum += row_accuracy
        # average_accuracy = accuracy_sum / len(sizes_series)
    return average_accuracy
# End of overall_accuracy()

def timestep_accuracy(masked_predictions, timestep_lengths):
    """
    Calculates the prediction accuracy for every timestep.
    Where the accuracy is NaN, the accuracy is replaced with 0. This should only happen in epochs where the given
    calculation is not done (eg. test_accuracy_op during training)

    Params:
    masked_predictions (tf.Tensor): The correct predictions, masked such that only valid predictions are present
    timestep_lengths (tf.Tensor): The number of possible valid predictions at each timestep

    Return:
    timestep_accuracies (tf.Tensor): The average accuracy for each timestep
    """
    with tf.variable_scope(constants.TIMESTEP_ACCURACY):
        timestep_predictions = tf.reduce_sum(masked_predictions, axis=0, name="sum_correct_predictions")
        timestep_accuracies = tf.divide(timestep_predictions, timestep_lengths, name="timestep_accuracies")
        timestep_accuracies = tf.where(tf.is_nan(timestep_accuracies), timestep_lengths, timestep_accuracies)
        timestep_accuracies = tf.unstack(timestep_accuracies, name="unstack_timestep_accuracies")
    return timestep_accuracies
# End of timestep_accuracy()
