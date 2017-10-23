"""
Contains functions for setting up the performance evaluation layer for a tensorflow-based RNN.

Copyright (c) 2017 Frank Derry Wanye

Date: 22 October, 2017
"""
import tensorflow as tf

from .. import constants

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