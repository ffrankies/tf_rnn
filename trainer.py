"""
Tensorflow implementation of a training method to train a given model.

Copyright (c) 2017 Frank Derry Wanye

Date: 30 September, 2017
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
    model (rnn.RNNModel): The model to train
    """
    model.logger.info("Started training the model.")
    # writer = tf.summary.FileWriter(model.model_path + "tensorboard", graph=model.session.graph)
    loss_list = []

    current_state = np.zeros((model.settings.train.batch_size, model.settings.rnn.hidden_size), dtype=float)
    for epoch_num in range(1, model.settings.train.epochs + 1):
        average_loss, current_state = train_epoch(model, epoch_num, current_state)
        loss_list.append(average_loss)
        # End of epoch training

    model.logger.info("Finished training the model. Final loss: %f" % average_loss)
    plot(model, loss_list)
# End of train()

def train_epoch(model, epoch_num, current_state):
    """
    Trains one full epoch.

    :type model: RNNModel()
    :param model: the model to train.

    :type epoch_num: int
    :param epoch_num: the number of the current epoch.

    :type current_state: numpy matrix
    :param current_state: the current hidden state.

    :type return: (float, numpy matrix)
    :param return: (the average incurred lost, the latest hidden state)
    """
    model.logger.info("Starting epoch: %d" % (epoch_num))

    total_loss = 0
    for batch_num in range(model.num_batches):
        # Debug log outside of function to reduce number of arguments.
        model.logger.debug("Training minibatch : ", batch_num, " | ", "epoch : ", epoch_num + 1)
        minibatch_loss, current_state = train_minibatch(model, batch_num, current_state)
        total_loss += minibatch_loss
    # End of batch training
    average_loss = total_loss / model.num_batches
    model.logger.info("Finished epoch: %d | loss: %f" % (epoch_num, average_loss))
    return average_loss, current_state
# End of train_epoch()

def train_minibatch(model, batch_num, current_state):
    """
    Trains one minibatch.

    Params:
    model (rnn.RNNModel): The model to train
    batch_num (int): The current batch number
    current_state (np.ndarray): The current hidden state of the model
    
    Return:
    tuple: (minibatch_loss, updated_hidden_state)
    """
    batch_x = model.inputs[batch_num]
    batch_y = model.labels[batch_num]
    sizes = model.sizes[batch_num]

    if batch_x[0][0] == model.token_to_index[constants.START_TOKEN]: # Reset state if start of sentence
        current_state = np.zeros((model.settings.train.batch_size, model.settings.rnn.hidden_size), dtype=float)
    
    total_loss, train_step, current_state, summary = model.session.run(
        [model.total_loss_op, model.train_step_fun, model.current_state, model.summary_ops],
        feed_dict={
            model.batch_x_placeholder:batch_x, 
            model.batch_y_placeholder:batch_y, 
            model.batch_sizes:sizes,
            model.hidden_state_placeholder:current_state
        })

    model.summary_writer.add_summary(summary)

    return total_loss, current_state
# End of train_minibatch()

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