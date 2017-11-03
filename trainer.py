"""
Tensorflow implementation of a training method to train a given model.

Copyright (c) 2017 Frank Derry Wanye

Date: 31 October, 2017
"""

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from . import constants

def train(model):
    """
    Trains the given model on the given dataset, and saves the losses incurred
    at the end of each epoch to a plot image. Also saves tensorflow event logs
    to the <model_path>/tensorboard directory for tensorboard functionality.

    Params:
    model (model.RNNModel): The model to train
    """
    model.logger.info("Started training the model.")
    # writer = tf.summary.FileWriter(model.model_path + "tensorboard", graph=model.session.graph)
    loss_list = []

    for epoch_num in range(1, model.settings.train.epochs + 1):
        average_loss = train_epoch(model, epoch_num)
        loss_list.append(average_loss)
        # End of epoch training

    test_loss = test_step(model)

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
    latest_state (np.ndarray): The latest state of the hidden layer
    """
    model.logger.info("Starting epoch: %d" % (epoch_num))

    cross_validation_loss = 0
    current_state = np.zeros(tuple(model.hidden_state_shape), dtype=float)
    for section in range(model.dataset.num_sections):
        model.dataset.next_iteration()
        train_step(model, epoch_num, current_state)
        minibatch_loss = validation_step(model, epoch_num, current_state)
        cross_validation_loss += minibatch_loss

    cross_validation_loss /= model.dataset.num_sections
    model.logger.info("Finished epoch: %d | loss: %f" % (epoch_num, cross_validation_loss))
    return cross_validation_loss
# End of train_epoch()

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

def validation_step(model, epoch_num, current_state):
    """
    Performs performance calculations on the dataset's validation partition.

    Params:
    model (model.RNNModel): The model to train
    epoch_num (int): The number of the current epoch
    current_state (np.ndarray): The current state of the hidden layer

    Return:
    average_loss (float): The average loss over all minibatches in the validation partition
    """
    total_validation_loss = 0
    for batch_num in range(model.dataset.valid.num_batches):
        # Debug log outside of function to reduce number of arguments.
        model.logger.debug("Validating minibatch : ", batch_num, " | ", "epoch : ", epoch_num)
        minibatch_loss, current_state = validate_minibatch(model, batch_num, current_state)
        total_validation_loss += minibatch_loss
    average_validation_loss = total_validation_loss / model.dataset.valid.num_batches
    return average_validation_loss
# End of validation_step()

def validate_minibatch(model, batch_num, current_state):
    """
    Calculates the performance of the network on one minibatch, logs the performance to tensorflow.

    Params:
    model (model.RNNModel): The model to validate
    batch_num (int): The current batch number
    epoch_num (int): The current epoch
    current_state (np.ndarray): The current hidden state of the model

    Return:
    minibatch_loss (float): The average loss over this minibatch
    updated_hidden_state (np.ndarray): The updated state of the hidden layer after validating
    """
    current_feed_dict = get_feed_dict(model, model.dataset.valid, batch_num, current_state)

    total_loss, current_state, summary = model.session.run(
        [model.total_loss_op, model.current_state, model.summary_ops],
        feed_dict=current_feed_dict
        )

    model.summary_writer.add_summary(summary)
    return total_loss, current_state
# End of validate_minibatch()

def test_step(model):
    """
    Finds the performance of the trained model on the testing partition of the dataset. Used as the definitive
    performance test for the model.

    Params:
    model (model.RNNModel): The trained model

    Return:
    averate_test_loss (float): The average loss on the testing partition
    """
    current_state = np.zeros(tuple(model.hidden_state_shape), dtype=float)
    total_test_loss = 0
    for batch_num in range(model.dataset.test.num_batches):
        # Debug log outside of function to reduce number of arguments.
        model.logger.debug("Testing minibatch : ", batch_num)
        minibatch_loss, current_state = test_minibatch(model, batch_num, current_state)
        total_test_loss += minibatch_loss
    average_test_loss = total_test_loss / model.dataset.test.num_batches
    return average_test_loss
# End of test_step()

def test_minibatch(model, batch_num, current_state):
    """
    Finds the average loss of a given minibatch for a trained network.

    Params:
    model (model.RNNModel): The model to test
    batch_num (int): The current batch number
    current_state (np.ndarray): The current hidden state of the model

    Return:
    minibatch_loss (float): The loss for this minibatch
    updated_state (np.ndarray): The updated hidden state of the model
    """
    current_feed_dict = get_feed_dict(model, model.dataset.test, batch_num, current_state)

    total_loss, current_state = model.session.run(
        [model.total_loss_op, model.current_state],
        feed_dict=current_feed_dict
        )

    return total_loss, current_state
# End of test_minibatch()

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
    batch = model.dataset.test.get_batch(batch_num)
    beginning = model.dataset.test.beginning[batch_num]
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
    Plots a graph of epochs against losses. Saves the plot to file in <model_path>/graph.png.

    :type model: RNNModel()
    :param model: the model whose loss graph will be plotted.

    :type loss_list: list()
    :param loss_list: the losses incurred during training.
    """
    plt.plot(range(1, len(loss_list) + 1), loss_list)
    plt.savefig(model.model_path + model.run_dir + constants.PLOT)
    plt.show()
# End of plot()
