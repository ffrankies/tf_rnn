"""
Tensorflow implementation of a training method to train a given model.

Copyright (c) 2017 Frank Derry Wanye

Date: 26 October, 2017
"""

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from . import constants
from .model import RNNModel

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

    model.logger.info("Finished training the model. Final loss: %f" % average_loss)
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
    current_state = np.zeros((model.settings.train.batch_size, model.settings.rnn.hidden_size), dtype=float)
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
    batch_x = model.dataset.train.x[batch_num]
    batch_y = model.dataset.train.y[batch_num]
    sizes = model.dataset.train.sizes[batch_num]

    if batch_x[0][0] == model.dataset.token_to_index[constants.START_TOKEN]: # Reset state if start of example
        current_state = np.zeros(tuple(model.hidden_state_shape), dtype=float)

    train_step, current_state = model.session.run(
        [model.train_step_fun, model.current_state],
        feed_dict={
            model.batch_x_placeholder:batch_x,
            model.batch_y_placeholder:batch_y,
            model.batch_sizes:sizes,
            model.hidden_state_placeholder:current_state
        })

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
    model (model.RNNModel): The model to train
    batch_num (int): The current batch number
    current_state (np.ndarray): The current hidden state of the model

    Return:
    minibatch_loss (float): The average loss over this minibatch
    updated_hidden_state (np.ndarray): The updated state of the hidden layer after training
    """
    batch_x = model.dataset.train.x[batch_num]
    batch_y = model.dataset.train.y[batch_num]
    sizes = model.dataset.train.sizes[batch_num]

    if batch_x[0][0] == model.dataset.token_to_index[constants.START_TOKEN]: # Reset state if start of example
        current_state = np.zeros(tuple(model.hidden_state_shape), dtype=float)

    total_loss, current_state, summary = model.session.run(
        [model.total_loss_op, model.current_state, model.summary_ops],
        feed_dict={
            model.batch_x_placeholder:batch_x,
            model.batch_y_placeholder:batch_y,
            model.batch_sizes:sizes,
            model.hidden_state_placeholder:current_state
        })

    model.summary_writer.add_summary(summary)
    return total_loss, current_state
# End of validate_minibatch()

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
