"""Tensorflow implementation of a training method to train a given model.

@since 0.6.1
"""
import math
import os

import dill
import numpy as np

from . import plotter
from . import constants

from .logger import info, debug, trace

# Only imported for type hints
from .batchmaker import Batch
from .model import RNNBase
from .observer import Observer
from .dataset import DatasetBase, DataPartition
from .layers.utils import Accumulator, AccumulatorData


@info('Training the model')
@debug()
def train(model: RNNBase):
    """Trains the given model on the given dataset, and saves the losses incurred
    at the end of each epoch to a plot image. Also saves tensorflow event logs
    to the <model_path>/tensorboard directory for tensorboard functionality.

    Params:
    - model (model.RNNBase): The model to train
    """
    # Create accumulators, pass them to the training, validation and testing steps
    metrics = model.saver.meta.latest()[constants.METRICS]
    Observer.init(model.dataset.valid, model.settings.train.num_sequences_to_observe, model.run_dir)
    final_epoch = model.settings.train.epochs + 1
    for epoch_num in range(model.saver.meta.latest()[constants.EPOCH]+1, final_epoch):
        Observer.set_epoch(epoch_num)
        train_epoch(model, epoch_num, metrics.train, metrics.valid)
        model.saver.save_model(model, [epoch_num, metrics], metrics.valid.is_best_accuracy)
        if early_stop(metrics.valid, epoch_num, model.settings.train.epochs):
            final_epoch = epoch_num
            model.logger.info('Stopping early because validation partition no longer showing improvement')
            break
        plotter.plot(model, metrics.train, metrics.valid, metrics.test)
        # End of epoch training

    performance_eval(model, metrics.test)
    model.saver.save_model(model, [final_epoch, metrics], metrics.valid.is_best_accuracy)

    model.logger.info("Finished training the model. Final validation loss: %f | Final test loss: %f | "
                      "Final test accuracy: %f" %
                      (metrics.valid.losses[-1], metrics.test.losses[-1], metrics.test.accuracies()[-1]))
    plotter.plot(model, metrics.train, metrics.valid, metrics.test)
# End of train()


def train_epoch(model: RNNBase, epoch_num: int, train_accumulator: Accumulator, valid_accumulator: Accumulator):
    """Trains one full epoch.

    Params:
    - model (model.RNNBase): The model to train
    - epoch_num (int): The number of the current epoch
    - train_accumulator (layers.utils.Accumulator): The accumulator for training performance
    - valid_accumulator (layers.utils.Accumulator): The accumulator for validation performance
    """
    model.logger.info("Starting epoch: %d" % (epoch_num))

    current_state = np.zeros(tuple(model.hidden_state_shape), dtype=float)
    train_step(model, epoch_num, current_state, train_accumulator)
    validation_step(model, current_state, valid_accumulator)

    summarize(model, train_accumulator, valid_accumulator, epoch_num)
    log_post_epoch(model, epoch_num, train_accumulator, valid_accumulator)

    train_accumulator.next_epoch()
    valid_accumulator.next_epoch()
# End of train_epoch()


@debug()
def train_step(model: RNNBase, epoch_num: int, current_state: np.array, accumulator: Accumulator):
    """
    Trains the model on the dataset's training partition.

    Params:
    - model (model.RNNBase): The model to train
    - epoch_num (int): The number of the current epoch
    - current_state (np.array): The current state of the hidden layer
    - accumulator (layers.utils.Accumulator): The accumulator for training performance
    """
    for batch in model.dataset.train:
        if epoch_num == 0:
            validate_minibatch(model, batch, current_state, accumulator)
        else:
            current_state = train_minibatch(model, batch, current_state, accumulator)
# End of train_step()


@trace()
def train_minibatch(model: RNNBase, batch: Batch, current_state: np.array, accumulator: Accumulator) -> np.array:
    """Trains one minibatch.

    Params:
    - model (model.RNNBase): The model to train
    - batch (Batch): The batch to be used for training
    - current_state (np.array): The current hidden state of the model
    - accumulator (layers.utils.Accumulator): The accumulator for training performance

    Return:
    - updated_hidden_state (np.array): The updated state of the hidden layer after training
    """
    current_feed_dict = get_feed_dict(model, batch, current_state)
    performance_data, _, current_state = model.session.run(
        [model.performance_ops, model.train_step_fun, model.current_state],
        feed_dict=current_feed_dict)
    performance_data = list(performance_data)
    update_accumulator(accumulator, batch, performance_data)
    return current_state
# End of train_minibatch()


@trace()
def get_feed_dict(model: RNNBase, batch: Batch, current_state: np.array, training: bool = False) -> dict:
    """Obtains the information needed for running tensorflow operations as a feed dictionary.

    Params:
    - model (model.RNNBase): The model containing the operations
    - batch (Batch): The batch whose information will make up the feed dict
    - current_state (np.array): The current hidden state of the RNN
    - training (bool): Whether or not this batch is being trained on (default: False)

    Return:
    - feed_dict (dict): The dictionary holding the necessary information for running tensorflow operations
    """
    current_state = reset_state(current_state, batch.beginning)
    feed_dict = build_feed_dict(model, batch, current_state, training)
    return feed_dict
# End of get_feed_dict()


@trace()
def reset_state(current_state: np.array, beginning: bool) -> np.array:
    """Resets the current state to zeros if the batch contains data from the beginning of a sequence.

    Params:
    - current_state (np.array): The current hidden state of the network after training the previous batch
    - beginning (boolean): True if the batch represents the start of a sequence

    Return:
    - current_state (np.array): The current hidden state of the network.
    """
    if beginning:  # If start of sequence
        current_state = np.zeros_like(current_state)
    return current_state
# End of reset_state()


@trace()
def build_feed_dict(model: RNNBase, batch: Batch, current_state: np.array, training: bool = False) -> dict:
    """Builds a dictionary to feed into the model for performing tensorflow operations.

    Params:
    - model (model.RNNBase): The model for which to build the feed dictionary
    - batch (Batch): Contains the inputs, outputs and sequence lengths of the current batch
    - current_state (np.array): The current hidden state of the RNN
    - training (bool): Whether or not this batch is being trained on (default: False)

    Return:
    - feed_dict (dict): The dictionary built out of the provided batch and current state
    """
    if training:
        dropout = model.settings.rnn.dropout
    else:
        dropout = 1.0
    feed_dict = {
        model.batch_x_placeholder: batch.x,
        model.batch_y_placeholder: batch.y,
        model.batch_sizes: batch.sequence_lengths,
        model.hidden_state_placeholder: current_state,
        model.dropout: dropout
        }
    return feed_dict
# End of build_feed_dict()


@trace()
def update_accumulator(accumulator: Accumulator, batch: Batch, performance_data: list) -> AccumulatorData:
    """Adds a batch's performance data to the specified Accumulator object.

    Note: The Accumulator contains calculated inputs, but the rest of the data is obtained from
    the dataset.

    Params:
    - accumulator (layers.utils.Accumulator): The accumulator to update with new performance data
    - batch (Batch): The batch about which the performance data was gathered
    - performance_data (list): The performance data for the given minibatch
      - loss (float): The average loss for the given minibatch
      - size (int): The number of valid elements in this minibatch
      - timestep_accuracies (list): The average accuracy for each timestep in this minibatch
      - timestep_elements (list): The number of valid elements for each timestep in this minibatch
      - predictions (tf.Tensor): The predictions made at every timestep
      - labels (list): The labels for the minibatch
      - sequence_lengths (list): The list of lengths of each sequence in the minibatch
    """
    performance_data.extend([batch.y, batch.sequence_lengths])
    data = AccumulatorData._make(performance_data)
    accumulator.update(data=data, ending=batch.ending)
    return data
# End of update_accumulator()


@debug()
def validation_step(model: RNNBase, current_state: np.array, accumulator: Accumulator):
    """Performs performance calculations on the dataset's validation partition.

    Params:
    - model (model.RNNBase): The model to train
    - current_state (np.array): The current state of the hidden layer
    - accumulator (layers.utils.Accumulator): The accumulator for validation performance
    """
    for batch in model.dataset.valid:
        current_state = validate_minibatch(model, batch, current_state, accumulator)
# End of validation_step()


@trace()
def validate_minibatch(model: RNNBase, batch: Batch, current_state: np.array, accumulator: Accumulator) -> np.array:
    """Calculates the performance of the network on one minibatch, logs the performance to tensorflow.

    Params:
    - model (model.RNNBase): The model to validate
    - batch (Batch): The batch to be used for validation
    - current_state (np.array): The current hidden state of the model
    - accumulator (layers.utils.Accumulator): The accumulator for validation performance

    Return:
    - updated_hidden_state (np.array): The updated state of the hidden layer after validating
    """
    current_feed_dict = get_feed_dict(model, batch, current_state)
    performance_data, current_state = model.session.run(
        [model.performance_ops, model.current_state],
        feed_dict=current_feed_dict
        )
    performance_data = list(performance_data)
    accumulator_data = update_accumulator(accumulator, batch, performance_data)
    Observer.observe(accumulator_data, model.dataset.indexer)
    return current_state
# End of validate_minibatch()


@debug()
def summarize(model: RNNBase, train_accumulator: Accumulator, valid_accumulator: Accumulator, epoch_num: int):
    """Calculates the network's performance at the end of a particular epoch. Writes performance data to the
    summary.

    Params:
    - model (model.RNNBase): The RNN model containing the operations and placeholders for the performance calculations
    - train_accumulator (layers.utils.Accumulator): The accumulator for training performance
    - valid_accumulator (layers.utils.Accumulator): The accumulator for validation performance
    - epoch_num (int): The epoch number for which the calculation is performed
    """
    model.logger.debug('Evaluating the model\'s performance after training for an epoch')
    summary = model.session.run(
        [model.summary_ops],
        feed_dict={
            model.train_performance.average_loss: train_accumulator.loss,
            # model.train_performance.average_accuracy: train_accumulator.accuracy,
            model.train_performance.timestep_accuracies: train_accumulator.get_timestep_accuracies(),
            model.validation_performance.average_loss: valid_accumulator.loss,
            # model.validation_performance.average_accuracy: valid_accumulator.accuracy,
            model.validation_performance.timestep_accuracies: valid_accumulator.get_timestep_accuracies(),
        })
    model.summary_writer.add_summary(summary[0], epoch_num)
# End of summarize()


@info('Performing a performance evaluation after training')
def performance_eval(model: RNNBase, accumulator: Accumulator):
    """Performs a final performance evaluation on the test partition of the dataset.

    Params:
    - model (model.RNNBase): The RNN model containing the session and tensorflow variable placeholders
    - accumulator (layers.utils.Accumulator): Accumulator for performance metrics of the test dataset
            partition
    """
    test_step(model, accumulator)
    # summary = model.session.run(
    #     [model.summary_ops],
    #     feed_dict={
    #         model.test_performance.average_loss: accumulator.loss,
    #         model.test_performance.average_accuracy: accumulator.accuracy,
    #         model.test_performance.timestep_accuracies: accumulator.timestep_accuracies
    #     })
    # model.summary_writer.add_summary(summary[0], epoch_num)
    accumulator.next_epoch()
# End of test_final()


@debug('Evaluating model\'s performance on test partition')
def test_step(model: RNNBase, accumulator: Accumulator):
    """Finds the performance of the trained model on the testing partition of the dataset. Used as the definitive
    performance test for the model.

    Params:
    - model (model.RNNBase): The trained model
    - accumulator (layers.utils.Accumulator): Accumulator for performance metrics of the test dataset
                partition
    """
    current_state = np.zeros(tuple(model.hidden_state_shape), dtype=float)
    for batch in model.dataset.test:
        current_state = test_minibatch(model, batch, current_state, accumulator)
# End of test_step()


@trace()
def test_minibatch(model: RNNBase, batch: Batch, current_state: np.array, accumulator: Accumulator) -> np.array:
    """Calculates the performance of the network on one minibatch, and saves the .

    Params:
    - model (model.RNNBase): The model to validate
    - batch (Batch): The batch to be used for testing the model
    - current_state (np.array): The current hidden state of the model
    - accumulator (layers.utils.Accumulator): The accumulator for validation performance

    Return:
    - updated_hidden_state (np.array): The updated state of the hidden layer after validating
    """
    current_feed_dict = get_feed_dict(model, batch, current_state)
    performance_data, current_state = model.session.run(
        [model.performance_ops, model.current_state],
        feed_dict=current_feed_dict
        )
    performance_data = list(performance_data)
    data = update_accumulator(accumulator, batch, performance_data)
    save_predictions(data.predictions, model, batch)
    return current_state
# End of test_minibatch()


@trace()
def early_stop(valid_accumulator: Accumulator, epoch_num: int, num_epochs: int) -> bool:
    """Checks whether or not the model should stop training because it has started to over-fit the data.

    Params:
    - valid_accumulator (layers.utils.Accumulator): The accumulator for validation performance
    - epoch_num (int): The number of epochs the model has trained for
    - num_epochs (int): The maximum number of epochs the model is set to train for

    Return:
    - should_stop (bool): True if the model has started to overfit the data
    """
    should_stop = False
    portion_size = min(10, max(5, math.ceil(num_epochs/10)))  # Keep portion size between 5 and 10
    if epoch_num >= portion_size:
        maximum = max(valid_accumulator.accuracies()[-portion_size:])
        last_accuracy = valid_accumulator.accuracies()[-1]
        if maximum < valid_accumulator.best_accuracy() and last_accuracy < valid_accumulator.best_accuracy() * 0.98:
            should_stop = True
    return should_stop
# End of early_stop


@trace()
def save_predictions(predictions: np.array, model: RNNBase, batch: Batch):
    """Saves the predictions made by the RNN during testing for analysis.

    Params:
    - data (np.array): The predictions made
    - model (RNNBase): Contains the test partition and saver object
    - batch (Batch): The batch for which the predictions were made
    """
    predictions_path = model.saver.meta.latest()[constants.DIR] + 'predictions.csv'
    if not os.path.isfile(predictions_path):
        with open(predictions_path, 'w') as predictions_file:
            predictions_file.write('sequence | prediction | ending?\n')
    sequences = [x + [y[-1]] for x, y in zip(batch.x, batch.y)]
    with open(predictions_path, 'a') as predictions_file:
        for sequence, prediction in zip(sequences, predictions):
            predictions_file.write("{} | {} | {}\n".format(sequence, prediction, batch.ending))
# End of save_predictions()


def log_post_epoch(model: RNNBase, epoch_num: int, train_accumulator: Accumulator, valid_accumulator: Accumulator):
    """Logs the epoch number, training loss, and validation loss and accuracy. If it's the first epoch (epoch 0), then
    only logs epoch number and training loss.

    Params:
    - model (RNNBase): The RNN model doing the logging
    - epoch_num (int): The number of the current epoch
    - train_accumulator (layers.utils.Accumulator): The accumulator for training performance
    - valid_accumulator (layers.utils.Accumulator): The accumulator for validation performance
    """
    if epoch_num == 0:
        model.logger.info("Finished epoch: {:d} | training_loss: {:.2f}".format(epoch_num, train_accumulator.loss))
    else:
        model.logger.info("Finished epoch: {:d} | training_loss: {:.2f} | validation_loss: {:.2f} | "
                          "validation_accuracy: {:.2f}".format(epoch_num, train_accumulator.loss, 
                          valid_accumulator.loss, valid_accumulator.accuracies()[-1]))
# End of log_post_epoch()
