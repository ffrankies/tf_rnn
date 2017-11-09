"""
Tensorflow implementation of a training method to train a given model.

Copyright (c) 2017 Frank Derry Wanye

Date: 9 November, 2017
"""

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from . import constants
from .layers.performance_layer import *

def train(model):
    """
    Trains the given model on the given dataset, and saves the losses incurred
    at the end of each epoch to a plot image. Also saves tensorflow event logs
    to the <model_path>/tensorboard directory for tensorboard functionality.

    Params:
    model (model.RNNModel): The model to train
    """
    model.logger.info("Started training the model.")
    loss_list = []

    for epoch_num in range(model.settings.train.epochs + 1):
        average_loss = train_epoch(model, epoch_num)
        loss_list.append(average_loss)
        # End of epoch training

    test_loss = performance_eval(model)

    model.logger.info("Finished training the model. Final validation loss: %f. Final test loss: %f" %
            (average_loss, test_loss))
    plot(model, loss_list)
# End of train()

def train_epoch(model, epoch_num):
    """
    Trains one full epoch.

    Params:
    model (model.RNNModel): The model to train
    epoch_num (int): The number of the current epoch

    Return:
    average_loss (float): The average incurred loss
    """
    model.logger.info("Starting epoch: %d" % (epoch_num))

    current_state = np.zeros(tuple(model.hidden_state_shape), dtype=float)
    validation_variables = create_performance_variables(model.dataset, "loss")
    for section in range(model.dataset.num_sections):
        model.dataset.next_iteration()
        if epoch_num != 0: train_step(model, epoch_num, current_state)
        validation_step(model, epoch_num, current_state, validation_variables)
    validation_variables.complete()
    cross_validation_loss = intermediate_performance(model, validation_variables, epoch_num)

    model.logger.info("Finished epoch: %d | loss: %f" % (epoch_num, cross_validation_loss))
    return cross_validation_loss
# End of train_epoch()

def create_performance_variables(dataset, mode):
    """
    Sets up the PerformanceVariables using a given dataset.

    Params:
    dataset (dataset.Dataset): The dataset from which the PerformanceVariables object will be built
    mode (string): Can be either "loss" or "accuracy" - denotes whether the object will store data needed to calculate
                   loss or accuracy

    Return:
    variables (layers.performance_layer.PerformanceVariables): The variables needed to calculate loss or accuracy
    """
    if mode not in ['loss', 'accuracy']:
        raise ValueError('The mode has to be one of "loss" and "accuracy".')
    max_length = dataset.max_length
    shapes = [np.shape(dataset.test.x[0]), np.shape(dataset.test.y[0])]
    if mode == 'loss':
        types = [np.float32, np.int32]
    else:
        types = [np.int32, np.int32]
    pad = dataset.token_to_index[constants.END_TOKEN]
    return PerformanceVariables(max_length, shapes, types, pad)
# End of create_performance_variables()

def train_step(model, epoch_num, current_state):
    """
    Trains the model on the dataset's training partition.

    Params:
    model (model.RNNModel): The model to train
    epoch_num (int): The number of the current epoch
    current_state (np.ndarray): The current state of the hidden layer
    """
    for batch_num in range(model.dataset.train.num_batches):
        # Debug log outside of function to reduce number of arguments.
        model.logger.debug("Training minibatch : ", batch_num, " | ", "epoch : ", epoch_num)
        current_state = train_minibatch(model, batch_num, current_state)
# End of train_step()

def train_minibatch(model, batch_num, current_state):
    """
    Trains one minibatch.

    Params:
    model (model.RNNModel): The model to train
    batch_num (int): The current batch number
    current_state (np.ndarray): The current hidden state of the model

    Return:
    updated_hidden_state (np.ndarray): The updated state of the hidden layer after training
    """
    current_feed_dict = get_feed_dict(model, model.dataset.train, batch_num, current_state)

    train_step, current_state = model.session.run(
        [model.train_step_fun, model.current_state],
        feed_dict=current_feed_dict)

    return current_state
# End of train_minibatch()

def validation_step(model, epoch_num, current_state, variables):
    """
    Performs performance calculations on the dataset's validation partition.

    Params:
    model (model.RNNModel): The model to train
    epoch_num (int): The number of the current epoch
    current_state (np.ndarray): The current state of the hidden layer

    Return:
    average_loss (float): The average loss over all minibatches in the validation partition
    """
    for batch_num in range(model.dataset.valid.num_batches):
        # Debug log outside of function to reduce number of arguments.
        model.logger.debug("Validating minibatch : ", batch_num, " | ", "epoch : ", epoch_num)
        current_state = validate_minibatch(model, model.dataset.valid, batch_num, current_state, variables)
# End of validation_step()

def validate_minibatch(model, dataset_partition, batch_num, current_state, variables):
    """
    Calculates the performance of the network on one minibatch, logs the performance to tensorflow.

    Params:
    model (model.RNNModel): The model to validate
    batch_num (int): The current batch number
    current_state (np.ndarray): The current hidden state of the model

    Return:
    minibatch_loss (float): The average loss over this minibatch
    updated_hidden_state (np.ndarray): The updated state of the hidden layer after validating
    """
    current_feed_dict = get_feed_dict(model, dataset_partition, batch_num, current_state)

    batch_logits, current_state = model.session.run(
        [model.logits_series, model.current_state],
        feed_dict=current_feed_dict
        )
    batch_logits = [row.tolist() for row in batch_logits]
    add_performance_batch(variables, dataset_partition, batch_num, batch_logits)

    return current_state
# End of validate_minibatch()

def add_performance_batch(variables, dataset_partition, batch_num, inputs):
    """
    Adds a batch to the specified PerformanceVariables object.
    Note: The PerformanceVariables container contains calculated inputs, but the rest of the data is obtained from
    the dataset.

    Params:
    variables (layers.performance_layer.PerformanceVariables): The container object to which to add the new batch
    dataset_partition (dataset.DatasetPartition): The dataset partition containing the remainder of the batch data
    batch_num (int): The index of the batch (within the dataset partition) that is being added to the container
    inputs (np.ndarray): The inputs to be added to the container
    """
    variables.add_batch(
        inputs=inputs,
        labels=dataset_partition.y[batch_num].tolist(),
        sizes=dataset_partition.sizes[batch_num],
        beginning=dataset_partition.beginning[batch_num],
        ending=dataset_partition.ending[batch_num])
# End of add_performance_batch()

def intermediate_performance(model, variables, epoch_num):
    """
    Calculates the network's performance at the end of a particular epoch. Writes performance data to the
    summary.

    Params:
    model (model.RNNModel): The RNN model containing the operations and placeholders for the performance calculations
    variables (layers.performance_layer.PerformanceVariables): The variables needed for the performance calculations
    epoch_num (int): The epoch number for which the calculation is performed

    Return:
    average_loss (float): The average loss incurred for this epoch
    """
    x = variables.inputs
    y = variables.labels
    s = variables.sizes

    epoch_loss, summary = model.session.run(
        [model.validation_loss_op, model.summary_ops],
        feed_dict={
            model.valid_logits:x,
            model.valid_labels:y,
            model.valid_sizes:s
        })

    model.summary_writer.add_summary(summary, epoch_num)

    return epoch_loss
# End of validate_epoch()

def get_test_performance_data(model):
    """
    Finds the performance of the trained model on the testing partition of the dataset. Used as the definitive
    performance test for the model.

    Params:
    model (model.RNNModel): The trained model

    Return:
    variables (layers.performance_layer.PerformanceVariables): Container object for the data needed to perform a loss
                                                               calculation on the test dataset partition
    """
    validation_variables = create_performance_variables(model.dataset, "loss")
    current_state = np.zeros(tuple(model.hidden_state_shape), dtype=float)
    total_test_loss = 0
    for batch_num in range(model.dataset.test.num_batches):
        # Debug log outside of function to reduce number of arguments.
        model.logger.debug("Testing minibatch : ", batch_num)
        current_state = validate_minibatch(model, model.dataset.test, batch_num, current_state, validation_variables)
    validation_variables.complete()
    return validation_variables
# End of test_step()

def performance_eval(model):
    """
    Performs a final performance evaluation on the test partition of the dataset.

    Params:
    model (model.RNNModel): The RNN model containing the session and tensorflow variable placeholders

    Return:
    loss (float): The calculated loss for the test partition of the dataset
    """
    variables = get_test_performance_data(model)
    x = variables.inputs
    y = variables.labels
    s = variables.sizes

    epoch_loss = model.session.run(
        [model.test_loss_op],
        feed_dict={
            model.test_logits:x,
            model.test_labels:y,
            model.test_sizes:s
        })

    return epoch_loss[0]
# End of test_final()

def get_feed_dict(model, dataset, batch_num, current_state):
    """
    Obtains the information needed for running tensorflow operations as a feed dictionary.

    Params:
    model (model.RNNModel): The model containing the operations
    dataset (dataset.DataPartition): The dataset from which to extract the batch information
    batch_num (int): The index of the batch in the dataset
    current_state (np.ndarray): The current hidden state of the RNN

    Return:
    feed_dict (dict): The dictionary holding the necessary information for running tensorflow operations
    """
    batch = dataset.get_batch(batch_num)
    beginning = dataset.beginning[batch_num]
    current_state = reset_state(current_state, beginning)
    feed_dict=build_feed_dict(model, batch, current_state)
    return feed_dict
# End of get_feed_dict()

def reset_state(current_state, beginning):
    """
    Resets the current state to zeros if the batch contains data from the beginning of a sequence.

    Params:
    current_state (np.ndarray): The current hidden state of the network after training the previous batch
    beginning (boolean): True if the batch represents the start of a sequence

    Return:
    current_state (np.ndarray): The current hidden state of the network.
    """
    if beginning is True: # If start of sequence
        current_state = np.zeros_like(current_state)
    return current_state
# End of reset_state()

def build_feed_dict(model, batch, current_state):
    """
    Builds a dictionary to feed into the model for performing tensorflow operations.

    Params:
    model (model.RNNModel): The model for which to build the feed dictionary
    batch (tuple): Contains the inputs, outputs and sizes of the current batch
    current_state (np.ndarray): The current hidden state of the RNN

    Return:
    feed_dict (dict): The dictionary built out of the provided batch and current state
    """
    x, y, sizes = batch
    feed_dict = {
        model.batch_x_placeholder:x,
        model.batch_y_placeholder:y,
        model.batch_sizes:sizes,
        model.hidden_state_placeholder:current_state}
    return feed_dict
# End of build_feed_dict()

def plot(model, loss_list):
    """
    Plots a grminibatch_loss, aph of epochs against losses. Saves the plot to file in <model_path>/graph.png.

    :type model: RNNModel()
    :param model: the model whose loss graph will be plotted.

    :type loss_list: list()
    :param loss_list: the losses incurred during training.
    """
    plt.plot(range(1, len(loss_list) + 1), loss_list)
    plt.savefig(model.model_path + model.run_dir + constants.PLOT)
    plt.show()
# End of plot()
