"""Contains functions for setting up the performance evaluation layer for a tensorflow-based RNN.

@since 0.6.1
"""

import tensorflow as tf
import numpy as np

from .. import constants


class PerformancePlaceholders(object):
    """Holds the placeholders for feeding in performance information for a given dataset partition.

    Instance variables:
    - average_loss (tf.placeholder_with_default): The average loss for the partition
    - average_accuracy (tf.placeholder_with_default): The average accuracy for the partition
    - timestep_accuracies (tf.placeholder_with_default): The average accuracy for each timestep for the partition
    """

    def __init__(self, max_timesteps: int):
        """
        Creates a new PerformancePlaceholders object
        """
        self.average_loss = tf.placeholder_with_default(input=0.0, shape=(), name='average_loss')
        self.average_accuracy = tf.placeholder_with_default(input=0.0, shape=(), name='average_accuracy')
        zero_accuracies = np.zeros([max_timesteps], np.float32)  # pylint: disable=E1101
        self.timestep_accuracies = tf.placeholder_with_default(input=zero_accuracies, shape=np.shape(zero_accuracies),
                                                               name='timestep_accuracies')
    # End of __init__()
# End of PerformancePlaceholders()


def performance_ops(logits_series, labels_series, sizes_series, truncate):
    """Performs all the performance calculations for a given minibatch

    Params:
    - logits_series (tf.Tensor): Calculated probabilities for each class for each input after training
    - labels_series (tf.Tensor): True labels for each input
    - sizes_series (tf.Tensor): The true, un-padded lengths of each row in the minibatch
    - truncate (int): The maximum sequence length for each minibatch

    Returns:
    - loss (float): The average loss for the given minibatch
    - accuracy (float): The average accuracy for the given minibatch
    - size (int): The number of valid elements in this minibatch
    - timestep_accuracies (list): The average accuracy for each timestep in this minibatch
    - timestep_elements (list): The number of valid elements for each timestep in this minibatch
    - predictions (tf.Tensor): The predictions made at every timestep
    """
    # calculate loss and accuracies for a minibatch
    avg_loss, batch_size = average_loss(logits_series, labels_series, sizes_series, truncate)
    avg_acc, timestep_accs, timestep_sizes, predictions = average_accuracy(
        logits_series, labels_series, sizes_series, truncate)
    return avg_loss, avg_acc, batch_size, timestep_accs, timestep_sizes, predictions
# End of performance_ops()


def average_loss(logits_series, labels_series, sizes_series, truncate):
    """
    Calculates the average loss for a given minibatch.

    Params:
    - logits_series (tf.Tensor): Calculated probabilities for each class for each input after training
    - labels_series (tf.Tensor): True labels for each input
    - sizes_series (tf.Tensor): The true, un-padded lengths of each row in the minibatch
    - truncate (int): The maximum sequence length for the minibatch

    Return:
    - loss (tf.Tensor): The average loss for this minibatch
    - size (tf.Tensor): The total number of elements in this minibatch
    """
    with tf.variable_scope(constants.LOSS_CALC):
        mask, _ = row_length_mask(sizes_series, truncate)  # Copied in here so that it can be used for training
        ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_series, labels=labels_series,
                                                                 name='ce_losses')
        ce_loss = tf.multiply(ce_loss, mask, name='mask_losses')  # Mask out invalid portions of the calculated losses
        total_batch_loss = tf.reduce_sum(ce_loss, axis=None, name='sum_losses')
        total_batch_size = tf.reduce_sum(sizes_series, axis=None, name='sum_sizes')
        total_batch_size = tf.cast(total_batch_size, dtype=tf.float32, name='cast_sizes_sum_to_float')
        loss = total_batch_loss / total_batch_size
    return loss, total_batch_size
# End of average_loss()


def average_accuracy(logits_series, labels_series, sizes_series, truncate):
    """
    Calculates the average accuracy for a given minibatch.

    Params:
    - logits_series (tf.Tensor): Calculated probabilities for each class for each input after training
    - labels_series (tf.Tensor): True labels for each input
    - sizes_series (tf.Tensor): The true, un-padded lengths of each row in the minibatch
    - truncate (int): The maximum sequence length for the minibatch

    Return:
    - accuracy (tf.Tensor): The average loss for this minibatch
    - timestep_accuracies (tf.Tensor): The average accuracy for each timestep in the given minibatch
    - timestep_sizes (tf.Tensor): The valid number of elements for each timestep in the given minibatch
    - predictions (tf.Tensor): The predictions made at every timestep
    """
    with tf.variable_scope(constants.ACCURACY):
        predictions, masked_predictions, timestep_lengths = predict_and_mask(
            logits_series, labels_series, sizes_series, truncate)
        avg_accuracy = overall_accuracy(masked_predictions, sizes_series)
        timestep_accuracies = timestep_accuracy(masked_predictions, timestep_lengths)
    return avg_accuracy, timestep_accuracies, timestep_lengths, predictions
# End of average_loss()


def predict_and_mask(logits_series, labels_series, sizes_series, max_row_length):
    """
    Finds the correct predictions made across the given logits, and applies a mask so that it only contains
    valid predictions.

    Params:
    - logits_series (tf.Tensor): Calculated probabilities for each class for each input
    - labels_series (tf.Tensor): True labels for each input
    - sizes_series (tf.Tensor): The true, un-padded lengths of each row in the minibatch
    - max_row_length (int): The maximum row length

    Return:
    - predictions (tf.Tensor): The predictions made at every timestep
    - masked_predictions (tf.Tensor): The correct predictions, after the mask has been applied to them
    - timestep_lengths (tf.Tensor): The number of valid predictions at each timestep
    """
    with tf.variable_scope(constants.PREDICTIONS_MASK):
        mask, timestep_lengths = row_length_mask(sizes_series, max_row_length)
        predictions = tf.nn.softmax(logits_series, dim=-1, name='logits_softmax')
        predictions = tf.argmax(predictions, axis=-1, name='logits_argmax', output_type=tf.int32)
        correct_predictions = tf.equal(predictions, labels_series)
        correct_predictions = tf.cast(correct_predictions, tf.float32)
        correct_predictions_masked = tf.multiply(correct_predictions, mask)
    return predictions, correct_predictions_masked, timestep_lengths
# End of predict_and_mask()


def row_length_mask(sizes_series, max_row_length):
    """
    Constructs a mask out of the row lengths series.

    Params:
    - sizes_series (tf.Tensor): The length of each sequence (row) in the data
    - max_row_length (int): The maximum length of sequences in the data

    Return:
    - mask (tf.Tensor): A mask containing 1s where the logits are valid, 0 where they are not
    - timestep_lengths (tf.Tensor): The number of valid logits at each timestep in the data
    """
    mask = tf.sequence_mask(sizes_series, maxlen=max_row_length, dtype=tf.float32, name='row_length_mask')
    timestep_lengths = tf.reduce_sum(mask, axis=0, name='timestep_lengths')
    return mask, timestep_lengths
# End of row_length_mask()


def overall_accuracy(masked_predictions, sizes_series):
    """
    Tensorflow operation that calculates the model's accuracy on a given minibatch.

    Params:
    - labels_series (tf.Tensor): True labels for each input
    - predictions_series (tf.Tensor): The predictions made by the RNN for each input

    Return:
    - average_accuracy (tf.Tensor): The average accuracy for each row in the minibatch
    """
    with tf.variable_scope(constants.ACCURACY):
        row_sums = tf.reduce_sum(masked_predictions, axis=1, name='correct_predictions_per_sequence')
        sizes_series = tf.cast(sizes_series, tf.float32, name='cast_row_lengths_to_float32')
        row_accuracies = tf.divide(row_sums, sizes_series, name='row_accuracies')
        row_accuracies = tf.where(tf.is_nan(row_accuracies), sizes_series, row_accuracies)
        accuracy = tf.reduce_mean(row_accuracies, name='average_accuracy')
    return accuracy
# End of overall_accuracy()


def timestep_accuracy(masked_predictions, timestep_lengths):
    """
    Calculates the prediction accuracy for every timestep.
    Where the accuracy is NaN, the accuracy is replaced with 0. This should only happen in epochs where the given
    calculation is not done (eg. test_accuracy_op during training)

    Params:
    - masked_predictions (tf.Tensor): The correct predictions, masked such that only valid predictions are present
    - timestep_lengths (tf.Tensor): The number of possible valid predictions at each timestep

    Return:
    - timestep_accuracies (tf.Tensor): The average accuracy for each timestep
    """
    with tf.variable_scope(constants.TIMESTEP_ACCURACY):
        timestep_predictions = tf.reduce_sum(masked_predictions, axis=0, name='sum_correct_predictions')
        timestep_accuracies = tf.divide(timestep_predictions, timestep_lengths, name='timestep_accuracies')
        timestep_accuracies = tf.where(tf.is_nan(timestep_accuracies), timestep_lengths, timestep_accuracies)
        timestep_accuracies = tf.unstack(timestep_accuracies, name='unstack_timestep_accuracies')
    return timestep_accuracies
# End of timestep_accuracy()
