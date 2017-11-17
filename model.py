'''
An RNN model implementation in tensorflow.

Copyright (c) 2017 Frank Derry Wanye

Date: 12 November, 2017
'''

import numpy as np
import tensorflow as tf
import logging
import ray
import time

from . import constants
from . import setup
from . import dataset
from . import saver
from . import tensorboard
from . import settings

from .layers.input_layer import *
from .layers.hidden_layer import *
# from .layers.output_layer import *
from .layers.performance_layer import *

class RNNModel(object):
    '''
    A basic RNN implementation in tensorflow.
    '''

    def __init__(self):
        '''
        Constructor for an RNN Model.
        '''
        self.settings = settings.Settings()
        model_path = saver.create_model_dir(self.settings.general.model_name)
        self.logger = setup.setup_logger(self.settings.logging, model_path)
        self.logger.info("RNN settings: %s" % self.settings)
        self.dataset = dataset.Dataset(self.logger, self.settings.rnn.dataset, self.settings.train)
        self.create_graph()
    # End of __init__()

    def create_graph(self):
        '''
        Creates all internal tensorflow operations and variables inside a local graph and session.
        '''
        self.logger.info('Creating the computational graph')
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.training()
            self.performance_evaluation()
            self.session = tf.Session(graph=self.graph)
            self.init_saver()
            self.session.run(tf.global_variables_initializer())
    # End of create_graph()

    def training(self):
        '''
        Creates tensorflow variables and operations needed for training.
        '''
        total_loss = self.loss_layer()
        with tf.variable_scope(constants.TRAINING):
            self.learning_rate = tf.Variable(
                initial_value=self.settings.train.learn_rate,
                dtype=tf.float32,
                name='learning_rate')
            self.train_step_fun = tf.train.AdagradOptimizer(self.learning_rate).minimize(total_loss)
    # End of training()

    def loss_layer(self):
        '''
        Evaluates the performance of the network on a given minibatch.
        Creates the following instance variables:
        - accuracy (tf.Tensor): The operation that calculates the average accuracy for the predictions on a given
                                minibatch
        - performance_ops (tf.Tensor): The operations that evaluate the performance of the network on a given minibatch

        Return:
        minibatch_loss_op (tf.Tensor): The operation that calculates the loss for the current minibatch
        '''
        logits_series = self.output_layer()
        with tf.variable_scope(constants.LOSS_LAYER):
            minibatch_loss_op, _ = average_loss(logits_series, self.batch_y_placeholder, self.batch_sizes,
                self.settings.train.truncate)
            self.performance_ops = performance_ops(logits_series, self.batch_y_placeholder, self.batch_sizes,
                self.settings.train.truncate)
        return minibatch_loss_op
    # End of performance_evaluation()

    def performance_evaluation(self):
        '''
        Creates variables for performance evaluation.

        Creates the following instance variables:
        - train_performance (layers.performance_layer.PerformancePlaceholders): placeholders for evaluating training
                                                                                performance
        - validation_performance (layers.performance_layer.PerformancePlaceholders): placeholders for evaluating
                                                                                     validation performance
        - test_performance (layers.performance_layer.PerformancePlaceholders): placeholders for evaluating test
                                                                               performance
        '''
        max_length = self.dataset.max_length
        with tf.variable_scope(constants.TRAINING_PERFORMANCE):
            self.train_performance = PerformancePlaceholders(max_length)
        with tf.variable_scope(constants.VALIDATION_PERFORMANCE):
            self.validation_performance = PerformancePlaceholders(max_length)
        with tf.variable_scope(constants.TEST_PERFORMANCE):
            self.test_performance = PerformancePlaceholders(max_length)
    # End of performance_evaluation()

    def output_layer(self):
        '''
        Creates the tensorflow variables and operations needed to compute the network outputs.
        Creates the following instance variables:
        - batch_y_placeholder (tf.placeholder): The placeholder for the labels
        - out_weights (tf.Variable): The output layer weights
        - out_bias (tf.Variable): The output layer bias
        - predictions_series (tf.Tensor): The predictions on a given minibatch

        Return:
        logits_series (tf.Tensor): The calculated probabilities of each class for each input in the minibatch
        '''
        states_series = self.hidden_layer()
        with tf.variable_scope(constants.OUTPUT):
            states_series = tf.unstack(states_series, axis=1, name='unstack_states_series')
            self.batch_y_placeholder = tf.placeholder(
                dtype=tf.int32,
                shape=np.shape(self.batch_x_placeholder),
                name='output_placeholder')
            self.out_weights = tf.Variable(
                initial_value=np.random.rand(self.settings.rnn.hidden_size, self.dataset.vocabulary_size),
                dtype=tf.float32,
                name='out_weights')
            self.out_bias = tf.Variable(
                np.zeros((self.dataset.vocabulary_size)),
                dtype=tf.float32,
                name='out_bias')
            logits_series = [
                tf.nn.xw_plus_b(state, self.out_weights, self.out_bias, name='state_times_out_weights')
                for state in states_series] #Broadcasted addition
            logits_series = tf.unstack(logits_series, axis=1, name='unstack_logits_series')
        with tf.variable_scope('predictions'):
            self.predictions_series = [tf.nn.softmax(logits) for logits in logits_series]
            logits_series = tf.stack(logits_series, axis=0, name='stack_logits_series')
        return logits_series
    # End of output_layer()

    def hidden_layer(self):
        '''
        Creates the tensorflow variables and operations needed to compute the hidden layer state.
        '''
        inputs_series = self.input_layer()
        with tf.variable_scope(constants.HIDDEN):
            self.batch_sizes = tf.placeholder(
                dtype=tf.int32,
                shape=[self.settings.train.batch_size],
                name='batch_sizes')
            hidden_state, self.hidden_state_placeholder, self.hidden_state_shape = layered_state_tuple(
                self.settings.rnn.layers, self.settings.train.batch_size, self.settings.rnn.hidden_size)
            cell = rnn_cell(self.settings.rnn.layers, self.settings.rnn.hidden_size, self.settings.rnn.dropout)
            states_series, self.current_state = tf.nn.dynamic_rnn(
                cell=cell,
                inputs=inputs_series,
                initial_state=hidden_state,
                sequence_length=self.batch_sizes)
        return states_series
    # End of hidden_layer()

    def input_layer(self):
        '''
        Creates the tensorflow variables and operations needed to perform the embedding lookup.
        '''
        with tf.variable_scope(constants.INPUT):
            self.batch_x_placeholder = tf.placeholder(
                dtype=tf.int32,
                shape=[self.settings.train.batch_size, self.settings.train.truncate],
                name='input_placeholder')
            if self.dataset.data_type == constants.TYPE_CHOICES[0]: # data type = 'text'
                inputs_series = token_to_vector(self.dataset.vocabulary_size, self.settings.rnn.hidden_size,
                    self.batch_x_placeholder)
            else:
                print('ERROR: Numeric inputs cannot be handled yet.')
                exit(-1)
        return inputs_series
    # End of input_layer()

    def init_saver(self):
        '''
        Creates the variables needed to save the model weights and tensorboard summaries.
        '''
        variables = ray.experimental.TensorFlowVariables(self.train_step_fun, self.session)
        self.saver = saver.Saver(self.logger, self.settings.general, variables, self.dataset.max_length)
        self.summary_writer, self.summary_ops = tensorboard.init_tensorboard(self)
    # End of init_saver()
# End of RNNModel()
