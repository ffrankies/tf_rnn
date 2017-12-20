'''
Tensorflow implementation of a training method to train a given model.
Copyright (c) 2017 Frank Derry Wanye
Date: 19 December, 2017
'''

import math
import numpy as np
import tensorflow as tf

from . import plotter
from . import constants
from .layers.performance_layer import Metrics
from .layers.performance_layer import Accumulator

def train(model):
    '''
    Trains the given model on the given dataset, and saves the losses incurred
    at the end of each epoch to a plot image. Also saves tensorflow event logs
    to the <model_path>/tensorboard directory for tensorboard functionality.
    
    Params:
    - model (model.RNNModel): The model to train
    '''
    model.logger.info('Started training the model.')

    # Create accumulators, pass them to the training, validation and testing steps
    metrics = model.saver.meta.latest()[constants.METRICS]

    for epoch_num in range(model.saver.meta.latest()[constants.EPOCH]+1, model.settings.train.epochs+1):
        train_epoch(model, epoch_num, metrics.train, metrics.valid)
        model.saver.save_model(model, [epoch_num, metrics], metrics.valid.is_best_accuracy)
        if early_stop(metrics.valid, epoch_num, model.settings.train.epochs) == True:
            model.logger.info('Stopping early because validation partition no longer showing improvement')
            break
        plotter.plot(model, metrics.train, metrics.valid, metrics.test)
        # End of epoch training

    performance_eval(model, epoch_num+1, metrics.test)

    model.logger.info("Finished training the model. Final validation loss: %f | Final test loss: %f | "
                      "Final test accuracy: %f" %
                      (metrics.valid.losses[-1], metrics.test.losses[-1], metrics.test.accuracies[-1]))
    plotter.plot(model, metrics.train, metrics.valid, metrics.test)
# End of train()

def train_epoch(model, epoch_num, train_accumulator, valid_accumulator):
    '''
    Trains one full epoch.
    
    Params:
    - model (model.RNNModel): The model to train
    - epoch_num (int): The number of the current epoch
    - train_accumulator (layers.performance_layer.Accumulator): The accumulator for training performance
    - valid_accumulator (layers.performance_layer.Accumulator): The accumulator for validation performance
    '''
    model.logger.info("Starting epoch: %d" % (epoch_num))

    current_state = np.zeros(tuple(model.hidden_state_shape), dtype=float)
    train_step(model, epoch_num, current_state, train_accumulator)
    validation_step(model, epoch_num, current_state, valid_accumulator)

    summarize(model, train_accumulator, valid_accumulator, epoch_num)
    model.logger.info("Finished epoch: %d | training_loss: %.2f | validation_loss: %.2f | validation_accuracy: %.2f" %
        (epoch_num, train_accumulator.loss, valid_accumulator.loss, valid_accumulator.accuracy))

    train_accumulator.next_epoch()
    valid_accumulator.next_epoch()
# End of train_epoch()

def train_step(model, epoch_num, current_state, accumulator):
    '''
    Trains the model on the dataset's training partition.
    
    Params:
    - model (model.RNNModel): The model to train
    - epoch_num (int): The number of the current epoch
    - current_state (np.ndarray): The current state of the hidden layer
    - accumulator (layers.performance_layer.Accumulator): The accumulator for training performance
    '''
    for batch_num in range(model.dataset.train.num_batches):
        # Debug log outside of function to reduce number of arguments.
        model.logger.debug("Training minibatch : %d | epoch : %d" % (batch_num, epoch_num))
        if epoch_num == 0:
            validate_minibatch(model, model.dataset.train, batch_num, current_state, accumulator)
        else:
            current_state = train_minibatch(model, batch_num, current_state, accumulator)
# End of train_step()

def train_minibatch(model, batch_num, current_state, accumulator):
    '''
    Trains one minibatch.
    
    Params:
    - model (model.RNNModel): The model to train
    - batch_num (int): The current batch number
    - current_state (np.ndarray): The current hidden state of the model
    - accumulator (layers.performance_layer.Accumulator): The accumulator for training performance
    
    Return:
    - updated_hidden_state (np.ndarray): The updated state of the hidden layer after training
    '''
    current_feed_dict = get_feed_dict(model, model.dataset.train, batch_num, current_state)
    performance_data, train_step, current_state = model.session.run(
        [model.performance_ops, model.train_step_fun, model.current_state],
        feed_dict=current_feed_dict)
    performance_data.extend([model.dataset.train.y[batch_num], model.dataset.train.sizes[batch_num]])
    update_accumulator(accumulator, model.dataset.train, batch_num, performance_data)
    return current_state
# End of train_minibatch()

def get_feed_dict(model, dataset, batch_num, current_state):
    '''
    Obtains the information needed for running tensorflow operations as a feed dictionary.
    
    Params:
    - model (model.RNNModel): The model containing the operations
    - dataset (dataset.DataPartition): The dataset from which to extract the batch information
    - batch_num (int): The index of the batch in the dataset
    - current_state (np.ndarray): The current hidden state of the RNN
    
    Return:
    - feed_dict (dict): The dictionary holding the necessary information for running tensorflow operations
    '''
    batch = dataset.get_batch(batch_num)
    beginning = dataset.beginning[batch_num]
    current_state = reset_state(current_state, beginning)
    feed_dict = build_feed_dict(model, batch, current_state)
    return feed_dict
# End of get_feed_dict()

def reset_state(current_state, beginning):
    '''
    Resets the current state to zeros if the batch contains data from the beginning of a sequence.
    
    Params:
    - current_state (np.ndarray): The current hidden state of the network after training the previous batch
    - beginning (boolean): True if the batch represents the start of a sequence
    
    Return:
    - current_state (np.ndarray): The current hidden state of the network.
    '''
    if beginning is True: # If start of sequence
        current_state = np.zeros_like(current_state)
    return current_state
# End of reset_state()

def build_feed_dict(model, batch, current_state):
    '''
    Builds a dictionary to feed into the model for performing tensorflow operations.
    
    Params:
    - model (model.RNNModel): The model for which to build the feed dictionary
    - batch (tuple): Contains the inputs, outputs and sizes of the current batch
    - current_state (np.ndarray): The current hidden state of the RNN
    
    Return:
    - feed_dict (dict): The dictionary built out of the provided batch and current state
    '''
    x, y, sizes = batch
    feed_dict = {
        model.batch_x_placeholder:x,
        model.batch_y_placeholder:y,
        model.batch_sizes:sizes,
        model.hidden_state_placeholder:current_state}
    return feed_dict
# End of build_feed_dict()

def update_accumulator(accumulator, dataset_partition, batch_num, performance_data):
    '''
    Adds a batch's performance data to the specified Accumulator object.
    
    Note: The Accumulator contains calculated inputs, but the rest of the data is obtained from
    the dataset.
    
    Params:
    - accumulator (layers.performance_layer.Accumulator): The accumulator to update with new performance data
    - dataset_partition (dataset.DatasetPartition): The dataset partition containing the remainder of the batch data
    - batch_num (int): The index of the batch (within the dataset partition) that is being added to the container
    - performance_data (list): The performance data for the given minibatch
      - loss (float): The average loss for the given minibatch
      - accuracy (float): The average accuracy for the given minibatch
      - size (int): The number of valid elements in this minibatch
      - timestep_accuracies (list): The average accuracy for each timestep in this minibatch
      - timestep_elements (list): The number of valid elements for each timestep in this minibatch
      - predictions (tf.Tensor): The predictions made at every timestep
    '''
    accumulator.update(
        data=performance_data,
        beginning=dataset_partition.beginning[batch_num],
        ending=dataset_partition.ending[batch_num])
# End of update_accumulator()

def validation_step(model, epoch_num, current_state, accumulator):
    '''
    Performs performance calculations on the dataset's validation partition.
    
    Params:
    - model (model.RNNModel): The model to train
    - epoch_num (int): The number of the current epoch
    - current_state (np.ndarray): The current state of the hidden layer
    - accumulator (layers.performance_layer.Accumulator): The accumulator for validation performance
    '''
    for batch_num in range(model.dataset.valid.num_batches):
        # Debug log outside of function to reduce number of arguments.
        model.logger.debug("Validating minibatch : %d | epoch : %d" % (batch_num, epoch_num))
        current_state = validate_minibatch(model, model.dataset.valid, batch_num, current_state, accumulator)
# End of validation_step()

def validate_minibatch(model, dataset_partition, batch_num, current_state, accumulator):
    '''
    Calculates the performance of the network on one minibatch, logs the performance to tensorflow.
    
    Params:
    - model (model.RNNModel): The model to validate
    - batch_num (int): The current batch number
    - current_state (np.ndarray): The current hidden state of the model
    - accumulator (layers.performance_layer.Accumulator): The accumulator for validation performance
    
    Return:
    - updated_hidden_state (np.ndarray): The updated state of the hidden layer after validating
    '''
    current_feed_dict = get_feed_dict(model, dataset_partition, batch_num, current_state)
    performance_data, current_state = model.session.run(
        [model.performance_ops, model.current_state],
        feed_dict=current_feed_dict
        )
    performance_data.extend([dataset_partition.y[batch_num], dataset_partition.sizes[batch_num]])
    update_accumulator(accumulator, dataset_partition, batch_num, performance_data)
    return current_state
# End of validate_minibatch()

def summarize(model, train_accumulator, valid_accumulator, epoch_num):
    '''
    Calculates the network's performance at the end of a particular epoch. Writes performance data to the
    summary.
    
    Params:
    - model (model.RNNModel): The RNN model containing the operations and placeholders for the performance calculations
    - train_accumulator (layers.performance_layer.Accumulator): The accumulator for training performance
    - valid_accumulator (layers.performance_layer.Accumulator): The accumulator for validation performance
    - epoch_num (int): The epoch number for which the calculation is performed
    '''
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
# End of summarize()

def performance_eval(model, epoch_num, accumulator):
    '''
    Performs a final performance evaluation on the test partition of the dataset.
    
    Params:
    - model (model.RNNModel): The RNN model containing the session and tensorflow variable placeholders
    - epoch_num (int): Total number of epochs + 1 (only used so that the performance summary shows up in tensorboard)
    - accumulator (layers.performance_layer.Accumulator): Accumulator for performance metrics of the test dataset
                                                        partition
    Return:
    - loss (float): The calculated loss for the test partition of the dataset
    - accuracy (float): The calculated accuracy for the test partition of the dataset
    - timestep_accuracies (list): The calculated accuracies for each timestep for the test partition of the dataset
    '''
    model.logger.debug('Performing a performance evaluation after training')
    test_step(model, accumulator)
    summary = model.session.run(
        [model.summary_ops],
        feed_dict={
            model.test_performance.average_loss:accumulator.loss,
            model.test_performance.average_accuracy:accumulator.accuracy,
            model.test_performance.timestep_accuracies:accumulator.timestep_accuracies
        })
    model.summary_writer.add_summary(summary[0], epoch_num)
    accumulator.next_epoch()
# End of test_final()

def test_step(model, accumulator):
    '''
    Finds the performance of the trained model on the testing partition of the dataset. Used as the definitive
    performance test for the model.
    
    Params:
    - model (model.RNNModel): The trained model
    - accumulator (layers.performance_layer.Accumulator): Accumulator for performance metrics of the test dataset
                partition
    '''
    model.logger.debug('Evaluating the model\'s performance on the test partition')
    current_state = np.zeros(tuple(model.hidden_state_shape), dtype=float)
    for batch_num in range(model.dataset.test.num_batches):
        # Debug log outside of function to reduce number of arguments.
        model.logger.debug("Testing minibatch : %d" % batch_num)
        current_state = validate_minibatch(model, model.dataset.test, batch_num, current_state, accumulator)
# End of test_step()

def early_stop(valid_accumulator, epoch_num, num_epochs):
    '''
    Checks whether or not the model should stop training because it has started to over-fit the data.

    Params:
    - valid_accumulator (layers.performance_layer.Accumulator): The accumulator for validation performance
    - epoch_num (int): The number of epochs the model has trained for
    - num_epochs (int): The maximum number of epochs the model is set to train for

    Return:
    - should_stop (boolean): True if the model has started to overfit the data
    '''
    should_stop = False
    one_tenth = math.ceil(num_epochs/10)
    if epoch_num >= one_tenth:
        maximum = max(valid_accumulator.accuracies[-one_tenth:])
        last_accuracy = valid_accumulator.accuracies[-1]
        if maximum < valid_accumulator.best_accuracy and last_accuracy < valid_accumulator.best_accuracy*0.98:
            should_stop = True
    return should_stop
# End of early_stop

def extend_performance_data(performance_data, index_to_token, dataset_partition, batch_num):
    '''
    Extends the performance data with the minibatch labels and sequence lengths. Also converts the predictions and 
    labels to tokens.

    Params:
    - performance_data (list): The list of performance data
      - loss (float): The average loss for the given minibatch
      - accuracy (float): The average accuracy for the given minibatch
      - size (int): The number of valid elements in this minibatch
      - timestep_accuracies (list): The average accuracy for each timestep in this minibatch
      - timestep_elements (list): The number of valid elements for each timestep in this minibatch
      - predictions (list): The predictions made at every timestep
    - index_to_token (list): Converts indexes to tokens in the vocabulary
    - dataset_partition (dataset.DatasetPartitioon): The dataset partition to which the minibatch belongs
    - batch_num (int): The index of the minibatch

    Returns:
    - extended_performance_data (list): The extended performance data
      - loss (float): The average loss for the given minibatch
      - accuracy (float): The average accuracy for the given minibatch
      - size (int): The number of valid elements in this minibatch
      - timestep_accuracies (list): The average accuracy for each timestep in this minibatch
      - timestep_elements (list): The number of valid elements for each timestep in this minibatch
      - predictions (list): The predictions made at every timestep, in token format
      - labels (list): The correct predictiosn for the minibatch
      - sequence_lengths (list): The lengths of each sequence in the minibatch 
    '''
    extended_performance_data = performance_data[:5]
    data_list = [performance_data[-1], dataset_partition.y[batch_num]]
    extended_performance_data.extend(indexes_to_tokens(index_to_token, data_list=data_list))
    extended_performance_data.append(dataset_partition.sizes[batch_num])
    return extended_performance_data
# End of extend_dataset_partition()

def indexes_to_tokens(index_to_token, data=None, data_list=None):
    '''
    Converts the indexes in the predictions and labels to the tokens they represent.

    Params:
    - index_to_token (list): Converts indexes to tokens
    - data (list): The data to be converted to token form
    - data_list (list): The list of data to be converted to token form

    Returns:
    - Either of the following:
      - tokenized_data (list): The data, in token form, when data is not None
      - tokenized_data_list (list): The list of data, in token form, when data is None
    '''
    if data is not None:
        token_data = [[index_to_token[item] for item in row] for row in data]
        return token_data
    elif data_list is not None:
        token_data_list = [index_to_token(index_to_token, data=item) for item in data_list]
        return token_data_list
    else:
        raise ValueError('No input provided to function.')
# End of indexes_to_tokens()