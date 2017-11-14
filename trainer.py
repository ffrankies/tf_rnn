"""
Tensorflow implementation of a training method to train a given model.

Copyright (c) 2017 Frank Derry Wanye

Date: 12 November, 2017
"""

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from . import constants
from .layers.performance_layer import Accumulator

def train(model):
    """
    Trains the given model on the given dataset, and saves the losses incurred
    at the end of each epoch to a plot image. Also saves tensorflow event logs
    to the <model_path>/tensorboard directory for tensorboard functionality.

    Params:
    model (model.RNNModel): The model to train
    """
    model.logger.info("Started training the model.")
    training_losses = []
    validation_losses = []

    for epoch_num in range(model.settings.train.epochs + 1):
        training_loss, validation_loss = train_epoch(model, epoch_num)
        training_losses.append(training_loss)
        validation_losses.append(validation_loss)
        # End of epoch training

    test_loss, test_accuracy, test_timestep_accuracy = performance_eval(model, epoch_num+1)

    model.logger.info("Finished training the model. Final validation loss: %f. Final test loss: %f"
                      "Final test accuracy: %f" % (validation_loss, test_loss, test_accuracy))
    plot(model, (training_losses, validation_losses), test_accuracy, test_timestep_accuracy)
# End of train()

def train_epoch(model, epoch_num):
    """
    Trains one full epoch.

    Params:
    model (model.RNNModel): The model to train
    epoch_num (int): The number of the current epoch

    Return:
    average_training_loss (float): The average incurred loss for training partition
    average_validation_loss (float): The average incurred loss for the validation partition
    """
    model.logger.info("Starting epoch: %d" % (epoch_num))

    current_state = np.zeros(tuple(model.hidden_state_shape), dtype=float)
    train_accumulator = Accumulator(model.logger, model.dataset.max_length)
    valid_accumulator = Accumulator(model.logger, model.dataset.max_length)
    for section in range(model.dataset.num_sections):
        model.dataset.next_iteration()
        train_step(model, epoch_num, current_state, train_accumulator)
        validation_step(model, epoch_num, current_state, valid_accumulator)

    log_intermediate_performance(model, train_accumulator, valid_accumulator, epoch_num)

    model.logger.info("Finished epoch: %d | training_loss: %f | validation_loss: %f" % 
        (epoch_num, train_accumulator.loss, valid_accumulator.loss))
    return train_accumulator.loss, valid_accumulator.loss
# End of train_epoch()

def train_step(model, epoch_num, current_state, accumulator):
    """
    Trains the model on the dataset's training partition.

    Params:
    model (model.RNNModel): The model to train
    epoch_num (int): The number of the current epoch
    current_state (np.ndarray): The current state of the hidden layer
    accumulator (layers.performance_layer.Accumulator): The accumulator for training performance
    """
    for batch_num in range(model.dataset.train.num_batches):
        # Debug log outside of function to reduce number of arguments.
        model.logger.debug("Training minibatch : %d | epoch : %d" % (batch_num, epoch_num))
        if epoch_num == 0:
            validate_minibatch(model, model.dataset.train, batch_num, current_state, accumulator)
        else:
            current_state = train_minibatch(model, batch_num, current_state, accumulator)
# End of train_step()

def train_minibatch(model, batch_num, current_state, accumulator):
    """
    Trains one minibatch.

    Params:
    model (model.RNNModel): The model to train
    batch_num (int): The current batch number
    current_state (np.ndarray): The current hidden state of the model
    accumulator (layers.performance_layer.Accumulator): The accumulator for training performance

    Return:
    updated_hidden_state (np.ndarray): The updated state of the hidden layer after training
    """
    current_feed_dict = get_feed_dict(model, model.dataset.train, batch_num, current_state)
    performance_data, train_step, current_state = model.session.run(
        [model.performance_ops, model.train_step_fun, model.current_state],
        feed_dict=current_feed_dict)
    update_accumulator(accumulator, model.dataset.train, batch_num, performance_data)
    return current_state
# End of train_minibatch()

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
    feed_dict = build_feed_dict(model, batch, current_state)
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

def validation_step(model, epoch_num, current_state, accumulator):
    """
    Performs performance calculations on the dataset's validation partition.

    Params:
    model (model.RNNModel): The model to train
    epoch_num (int): The number of the current epoch
    current_state (np.ndarray): The current state of the hidden layer
    accumulator (layers.performance_layer.Accumulator): The accumulator for validation performance
    """
    for batch_num in range(model.dataset.valid.num_batches):
        # Debug log outside of function to reduce number of arguments.
        model.logger.debug("Validating minibatch : %d | epoch : %d" % (batch_num, epoch_num))
        current_state = validate_minibatch(model, model.dataset.valid, batch_num, current_state, accumulator)
# End of validation_step()

def validate_minibatch(model, dataset_partition, batch_num, current_state, accumulator):
    """
    Calculates the performance of the network on one minibatch, logs the performance to tensorflow.

    Params:
    model (model.RNNModel): The model to validate
    batch_num (int): The current batch number
    current_state (np.ndarray): The current hidden state of the model
    accumulator (layers.performance_layer.Accumulator): The accumulator for validation performance

    Return:
    updated_hidden_state (np.ndarray): The updated state of the hidden layer after validating
    """
    current_feed_dict = get_feed_dict(model, dataset_partition, batch_num, current_state)
    performance_data, current_state = model.session.run(
        [model.performance_ops, model.current_state],
        feed_dict=current_feed_dict
        )
    update_accumulator(accumulator, dataset_partition, batch_num, performance_data)
    return current_state
# End of validate_minibatch()

def update_accumulator(accumulator, dataset_partition, batch_num, performance_data):
    """
    Adds a batch's performance data to the specified Accumulator object.
    Note: The Accumulator contains calculated inputs, but the rest of the data is obtained from
    the dataset.

    Params:
    accumulator (layers.performance_layer.Accumulator): The accumulator to update with new performance data
    dataset_partition (dataset.DatasetPartition): The dataset partition containing the remainder of the batch data
    batch_num (int): The index of the batch (within the dataset partition) that is being added to the container
    performance_data (list): The performance data for the given batch
    """
    accumulator.add_data(
        data=performance_data, 
        beginning=dataset_partition.beginning[batch_num],
        ending=dataset_partition.ending[batch_num])
# End of update_accumulator()

def log_intermediate_performance(model, train_accumulator, valid_accumulator, epoch_num):
    """
    Calculates the network's performance at the end of a particular epoch. Writes performance data to the
    summary.

    Params:
    model (model.RNNModel): The RNN model containing the operations and placeholders for the performance calculations
    train_accumulator (layers.performance_layer.Accumulator): The accumulator for training performance
    valid_accumulator (layers.performance_layer.Accumulator): The accumulator for validation performance
    epoch_num (int): The epoch number for which the calculation is performed
    """
    model.logger.debug('Evaluating the model\'s performance after training for an epoch')
    summary = model.session.run(
        [model.summary_ops],
        feed_dict={
            model.train_performance.average_loss:train_accumulator.loss,
            model.train_performance.average_accuracy:train_accumulator.accuracy,
            model.train_performance.timestep_accuracies:train_accumulator.timestep_accuracies,
            model.validation_performance.average_loss:valid_accumulator.loss,
            model.validation_performance.average_accuracy:valid_accumulator.accuracy,
            model.validation_performance.timestep_accuracies:valid_accumulator.timestep_accuracies,
        })
    model.summary_writer.add_summary(summary[0], epoch_num)
# End of log_intermediate_performance()

def test_step(model):
    """
    Finds the performance of the trained model on the testing partition of the dataset. Used as the definitive
    performance test for the model.

    Params:
    model (model.RNNModel): The trained model

    Return:
    variables (layers.performance_layer.PerformanceVariables): Container object for the data needed to perform a loss
                                                               calculation on the test dataset partition
    """
    model.logger.debug('Evaluating the model\'s performance on the test partition')
    test_accumulator = Accumulator(model.logger, model.dataset.max_length)
    current_state = np.zeros(tuple(model.hidden_state_shape), dtype=float)
    for batch_num in range(model.dataset.test.num_batches):
        # Debug log outside of function to reduce number of arguments.
        model.logger.debug("Testing minibatch : %d" % batch_num)
        current_state = validate_minibatch(model, model.dataset.test, batch_num, current_state, test_accumulator)
    return test_accumulator
# End of test_step()

def performance_eval(model, epoch_num):
    """
    Performs a final performance evaluation on the test partition of the dataset.

    Params:
    model (model.RNNModel): The RNN model containing the session and tensorflow variable placeholders
    epoch_num (int): Total number of epochs + 1 (only used so that the performance summary shows up in tensorboard)

    Return:
    loss (float): The calculated loss for the test partition of the dataset
    accuracy (float): The calculated accuracy for the test partition of the dataset
    timestep_accuracies (list): The calculated accuracies for each timestep for the test partition of the dataset
    """
    model.logger.debug('Performing a performance evaluation after training')
    test_accumulator = test_step(model)
    summary = model.session.run(
        [model.summary_ops],
        feed_dict={
            model.test_performance.average_loss:test_accumulator.loss,
            model.test_performance.average_accuracy:test_accumulator.accuracy,
            model.test_performance.timestep_accuracies:test_accumulator.timestep_accuracies
        })
    model.summary_writer.add_summary(summary[0], epoch_num)
    return test_accumulator.loss, test_accumulator.accuracy, test_accumulator.timestep_accuracies
# End of test_final()

def plot(model, loss_list, accuracy, timestep_accuracy):
    """
    Plots a graph of epochs against losses. Saves the plot to file in <model_path>/graph.png.

    Params:
    model (model.RNNModel): The model containing the path where the figure will be saved
    loss_list (list): The list of incurred losses
    accuracy (float): The average accuracy on the test dataset partition
    timestep_accuracy (list): The average accuracy for each timestep on the test dataset partition
    """
    model.logger.info("Plotting results for visualization")
    plt.figure(num=1, figsize=(10, 10))
    loss_axis = plt.subplot(311)
    plot_loss(loss_list, loss_axis)
    accuracy_axis = plt.subplot(312)
    plot_accuracy(accuracy, accuracy_axis)
    bar_chart_axis = plt.subplot(313)
    plot_bar_chart(timestep_accuracy, bar_chart_axis)
    plt.tight_layout()
    plt.savefig(model.model_path + model.run_dir + constants.PLOT)
    plt.show()
# End of plot()

def plot_loss(loss_list, axis):
    """
    Plots the training and validation losses on a sublot.

    Params:
    loss_list (tuple):
    - training_losses (list): The training losses to plot
    - validation_losses (list): The validation losses to plot
    axis (matplotlib.axes.Axes): The axis on which to plot the validation loss
    """
    x = range(0, len(loss_list[0]))
    axis.plot(x, loss_list[0], '-b', label='Training Loss (final=%.2f)' % loss_list[0][-1])
    axis.plot(x, loss_list[1], '-r', label='Validation Loss (final=%.2f)' % loss_list[1][-1])
    axis.legend(loc='upper right')
    axis.set_xlabel('Epoch')
    axis.set_ylabel('Average Loss')
    axis.set_title('Visualizing loss during training')
# End of plot_loss()

def plot_accuracy(accuracy, axis):
    """
    Plots the average accuracy on a separate subplot.

    Params:
    accuracy (float): The average accuracy of predictions for the test dataset partition
    axis (matplotlib.axes.Axes): The axis on which to plot the validation loss
    """
    accuracy = accuracy * 100
    axis.pie([accuracy, 100 - accuracy], labels=['Correct predictions', 'Incorrect predictions'], autopct='%.2f%%')
    axis.axis('equal')
    axis.set_title('Average accuracy for the test partition')
# End of plot_accuracy()

def plot_bar_chart(timestep_accuracy, axis):
    """
    Plots the average accuracy for each timestep as a bar chart on a separate subplot.

    Params:
    timestep_accuracy (list): The average accuracy of predictions for each timestep on the test dataset partition
    axis (matplotlib.axes.Axes): The axis on which to plot the validation loss
    """
    timestep_accuracy = [x * 100.0 for x in timestep_accuracy]
    bar_chart = axis.bar(range(1, len(timestep_accuracy) + 1), timestep_accuracy)
    axis.set_xlabel('Timestep')
    axis.set_ylabel('Average Accuracy (%)')
    axis.set_title('Average accuracy of each timestep for the test partition')
    # from https://matplotlib.org/gallery/api/barchart.html#sphx-glr-gallery-api-barchart-py
    # Couldn't find a better way to label the bar chart, unfortunately
    for bar, accuracy in zip(bar_chart, timestep_accuracy):
        x_pos = bar.get_x() + bar.get_width()/2.
        height = bar.get_height()
        y_pos = height - 20.0
        if height <= 30.0 : y_pos = height + 10.0
        axis.text(x_pos, y_pos, "%.1f" % accuracy, ha='center', va='bottom', rotation=90)
# End of plot_bar_chart()