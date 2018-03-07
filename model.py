"""
An RNN model implementation in tensorflow.

Copyright (c) 2017-2018 Frank Derry Wanye
@since 0.5.0
"""

import abc

import numpy as np
import tensorflow as tf
import ray  # pylint: disable=E0401

from . import constants
from . import dataset
from . import saver
from . import tensorboard
from . import settings
from . import utils

from .logger import Logger, info, debug
from .layers.input_layer import token_to_vector
from .layers.hidden_layer import layered_state_tuple, rnn_cell
# from .layers.output_layer import *
from .layers.performance_layer import average_loss, performance_ops, PerformancePlaceholders


def create_model(settings):
    """
    Creates a model of the given type.

    Params:
    - model_type (string): The type of the model to create
    """
    print(settings)
    if settings.rnn.num_features > 1 or len(settings.rnn.input_names) > 1:
        rnn_model = MultiFeatureRNN(model_settings=settings)
    else:
        rnn_model = BasicRNN(model_settings=settings)
    return rnn_model
# End of create_model()


class RNNBase(object):
    """The base for an RNN model, implemented in tensorflow.

    Instance variables:
    - settings (settings.Settings): The settings for the RNN
    - model_path (string): The path to the model's home directory
    - logger (logger.Logger): The logger for this RNN
    - dataset (dataset.SimpleDataset): The dataset on which to train the model
    - graph (tf.Graph): The model's computation graph
    - session (tf.Session): The model's session for tensorflow variable execution
    - learning_rate (tf.float32): The learning rate for the model
    - performance_ops (tf.Tensor): The operations that evaluate the performance of the network on a given minibatch
    - train_performance (layers.performance_layer.PerformancePlaceholders): placeholders for evaluating training
                performance
    - validation_performance (layers.performance_layer.PerformancePlaceholders): placeholders for evaluating
                validation performance
    - test_performance (layers.performance_layer.PerformancePlaceholders): placeholders for evaluating test
                performance
    - batch_y_placeholder (tf.placeholder): The placeholder for the labels
    - out_weights (tf.Variable): The output layer weights
    - out_bias (tf.Variable): The output layer bias
    - predictions_series (tf.Tensor): The predictions on a given minibatch
    - batch_sizes (tf.placeholder): The placeholder for the actual size of each sequence in the input minibatch
    - hidden_state_placeholder (tf.placeholder): The placeholder for the hidden state of the model
    - hidden_state_shape (tf.placeholder): The shape of the hidden state placeholder
    - dropout (tf.placeholder): The probability of keeping a weight when doing dropout
    - saver (saver.Saver): The object used for loading and saving the model's weights
    - run_dir (string): The directory in which the weights will be saved
    - summary_writer (tf.summary.FileWriter): The writer for the tensorboard events
    - summary_ops (tf.Tensor): The operations for the tensorboard summaries
    - variables (ray.experimental.TensorFlowVariables): All the tensorflow variables, for saving and loading
    """

    def __init__(self, model_settings: settings.Settings = None, model_dataset: dataset.DatasetBase = None):
        """Constructor for an RNN Model. Performs the following functions:
        - initializes model settings
        - initializes model path
        - initializes logger
        - loads and initializes the dataset
        - creates the computational graph

        Params:
        - model_settings (settings.Settings): The settings to be used for the model
        - model_dataset (dataset.DatasetBase): The dataset to be used for training the model
        """
        self.settings = settings.Settings() if model_settings is None else model_settings
        self.model_path = utils.create_model_dir(self.settings.general.model_name)
        self.logger = Logger(self.model_path)
        self.logger.info("RNN settings: %s" % self.settings)
        self.dataset = model_dataset
        if model_dataset is None:
            self.dataset = dataset.SimpleDataset(self.logger, self.settings.rnn, self.settings.train)
        self.create_graph()
    # End of __init__()

    @info('Creating the computational graph')
    def create_graph(self):
        """Creates all internal tensorflow operations and variables inside a local computational graph and session.
        """
        self.graph = tf.Graph()
        with self.graph.as_default():  # pylint: disable=E1129
            self.training()
            self.performance_evaluation()
            self.session = tf.Session(graph=self.graph)
            self.session.run(tf.global_variables_initializer())
            self.init_saver()
    # End of create_graph()

    @debug()
    def training(self):
        """Creates tensorflow variables and operations needed for training.
        """
        total_loss = self.loss_layer()
        with tf.variable_scope(constants.TRAINING):
            self.learning_rate = tf.Variable(
                initial_value=self.settings.train.learn_rate,
                dtype=tf.float32,
                name='learning_rate')
            self.train_step_fun = tf.train.AdagradOptimizer(self.learning_rate).minimize(total_loss)
    # End of training()

    @debug()
    def loss_layer(self) -> tf.Tensor:
        """Evaluates the performance of the network on a given minibatch.

        Returns:
        - minibatch_loss_op (tf.Tensor): The operation that calculates the loss for the current minibatch
        """
        logits_series = self.output_layer()
        with tf.variable_scope(constants.LOSS_LAYER):
            # row_lengths_series = tf.unstack(self.batch_sizes, name="unstack_batch_sizes")
            # labels_series = tf.unstack(self.batch_y_placeholder, name="unstack_labels_series")
            # self.accuracy = calculate_accuracy(labels_series, self.predictions_series)
            self.minibatch_loss_op, _ = average_loss(
                logits_series, self.batch_y_placeholder, self.batch_sizes, self.settings.train.truncate)
            self.performance_ops = performance_ops(
                logits_series, self.batch_y_placeholder, self.batch_sizes, self.settings.train.truncate)
        return self.minibatch_loss_op
    # End of performance_evaluation()

    @debug()
    def performance_evaluation(self):
        """Creates variables for performance evaluation.
        """
        max_length = self.dataset.max_length
        with tf.variable_scope(constants.TRAINING_PERFORMANCE):
            self.train_performance = PerformancePlaceholders(max_length)
        with tf.variable_scope(constants.VALIDATION_PERFORMANCE):
            self.validation_performance = PerformancePlaceholders(max_length)
        with tf.variable_scope(constants.TEST_PERFORMANCE):
            self.test_performance = PerformancePlaceholders(max_length)
    # End of performance_evaluation()

    @debug()
    def output_layer(self) -> tf.Tensor:
        """Creates the tensorflow variables and operations needed to compute the network outputs.

        Return:
        - logits_series (tf.Tensor): The calculated probabilities of each class for each input in the minibatch
        """
        states_series = self.hidden_layer()
        with tf.variable_scope(constants.OUTPUT):
            states_series = tf.unstack(states_series, axis=1, name="unstack_states_series")
            self.batch_y_placeholder = tf.placeholder(
                dtype=tf.int32,
                shape=np.shape(self.batch_x_placeholder),
                name="output_placeholder")
            self.out_weights = tf.Variable(
                initial_value=np.random.rand(self.settings.rnn.hidden_size, self.dataset.vocabulary_size),
                dtype=tf.float32,
                name="out_weights")
            self.out_bias = tf.Variable(
                np.zeros((self.dataset.vocabulary_size)),
                dtype=tf.float32,
                name="out_bias")
            logits_series = [
                tf.nn.xw_plus_b(state, self.out_weights, self.out_bias, name="state_times_out_weights")
                for state in states_series]  # Broadcasted addition
            logits_series = tf.unstack(logits_series, axis=1, name="unstack_logits_series")
        with tf.variable_scope("predictions"):
            self.predictions_series = [tf.nn.softmax(logits) for logits in logits_series]
            logits_series = tf.stack(logits_series, axis=0, name="stack_logits_series")
        return logits_series
    # End of output_layer()

    @debug()
    def hidden_layer(self) -> tf.Tensor:
        """Creates the tensorflow variables and operations needed to compute the hidden layer state.

        Return:
        - states_series (tf.tensor): The RNN state for each input for each timestep in the input
        """
        inputs_series = self.input_layer()
        with tf.variable_scope(constants.HIDDEN):
            self.batch_sizes = tf.placeholder(
                dtype=tf.int32,
                shape=[self.settings.train.batch_size],
                name='batch_sizes')
            hidden_state, self.hidden_state_placeholder, self.hidden_state_shape = layered_state_tuple(
                self.settings.rnn.layers, self.settings.train.batch_size, self.settings.rnn.hidden_size)
            self.dropout = tf.placeholder(
                dtype=tf.float32,
                shape=(),
                name='dropout')
            cell = rnn_cell(self.settings.rnn.layers, self.settings.rnn.hidden_size, self.dropout)
            states_series, self.current_state = tf.nn.dynamic_rnn(
                cell=cell,
                inputs=inputs_series,
                initial_state=hidden_state,
                sequence_length=self.batch_sizes)
        return states_series
    # End of hidden_layer()

    @abc.abstractmethod
    def input_layer(self) -> tf.Tensor:
        """
        Creates the tensorflow variables and operations needed to perform the embedding lookup.

        NOTE: This method **must** create a variable called self.batch_x_placeholder

        Returns:
        - inputs_series (tf.tensor): The inputs series for each timestep for each sequence in the inputs
        """
        self.batch_x_placeholder = None
    # End of input_layer()

    @debug()
    def init_saver(self):
        """Creates the variables needed to save the model weights and tensorboard summaries.
        Loads previous model if a new one wasn't requested.
        """
        self.saver = saver.Saver(self.logger, self.settings.general, self.dataset.max_length)
        self.variables = ray.experimental.TensorFlowVariables(self.minibatch_loss_op, self.session)
        if self.settings.general.new_model:
            self.saver.meta.increment_run(self.logger, self.dataset.max_length)
        else:
            self.saver.load_model(self, self.settings.general.best_model)
        self.run_dir = self.saver.meta.latest()[constants.DIR]
        self.summary_writer, self.summary_ops = tensorboard.init_tensorboard(self)
    # End of init_saver()
# End of RNNBase()


class BasicRNN(RNNBase):
    """A basic RNN implementation in tensorflow.

    @see RNNBase
    """

    @debug()
    def input_layer(self) -> tf.Tensor:
        """
        @see RNNBase.input_layer
        """
        with tf.variable_scope(constants.INPUT):
            self.batch_x_placeholder = tf.placeholder(
                dtype=tf.int32,
                shape=[self.settings.train.batch_size, self.settings.train.truncate],
                name='input_placeholder')
            if self.dataset.data_type == constants.TYPE_CHOICES[0]:  # data type = 'text'
                inputs_series = token_to_vector(
                    self.dataset.vocabulary_size,
                    self.settings.rnn.hidden_size,
                    self.batch_x_placeholder)
            else:
                print('ERROR: Numeric inputs cannot be handled yet.')
                exit(-1)
        return inputs_series
    # End of input_layer()
# End of BasicRNN


class MultiFeatureRNN(RNNBase):
    """
    An implementation of an RNN running with multiple inputs.

    @see RNNBase
    """

    @debug()
    def output_layer(self):
        """
        @see RNNBase.output_layer
        """
        states_series, hidden_size = self.hidden_layer()
        with tf.variable_scope(constants.OUTPUT):
            states_series = tf.unstack(states_series, axis=1, name='unstack_states_series')
            self.batch_y_placeholder = tf.placeholder(
                dtype=tf.int32,
                shape=[self.settings.train.batch_size, self.settings.train.truncate],
                name='output_placeholder')
            self.out_weights = tf.Variable(
                initial_value=np.random.rand(hidden_size, self.dataset.vocabulary_size[0]),
                dtype=tf.float32,
                name='out_weights')
            self.out_bias = tf.Variable(
                np.zeros((self.dataset.vocabulary_size[0])),
                dtype=tf.float32,
                name='out_bias')
            logits_series = [
                tf.nn.xw_plus_b(state, self.out_weights, self.out_bias, name='state_times_out_weights')
                for state in states_series]  # Broadcasted addition
            logits_series = tf.unstack(logits_series, axis=1, name='unstack_logits_series')
        with tf.variable_scope('predictions'):
            self.predictions_series = [tf.nn.softmax(logits) for logits in logits_series]
            logits_series = tf.stack(logits_series, axis=0, name='stack_logits_series')
        return logits_series
    # End of output_layer()

    @debug()
    def hidden_layer(self):
        """
        @see RNNBase.hidden_layer

        Also returns:
        - hidden_size (int): The size of the hidden layer for multiple inputs (hidden_size * number_of_inputs)
        """
        inputs_series = self.input_layer()
        with tf.variable_scope(constants.HIDDEN):
            self.batch_sizes = tf.placeholder(
                dtype=tf.int32,
                shape=[self.settings.train.batch_size],
                name='batch_sizes')
            hidden_size = self.settings.rnn.hidden_size * len(self.settings.rnn.input_names)
            hidden_state, self.hidden_state_placeholder, self.hidden_state_shape = layered_state_tuple(
                self.settings.rnn.layers, self.settings.train.batch_size, hidden_size)
            self.dropout = tf.placeholder(
                dtype=tf.float32,
                shape=(),
                name='dropout')
            cell = rnn_cell(self.settings.rnn.layers, hidden_size, self.dropout)
            states_series, self.current_state = tf.nn.dynamic_rnn(
                cell=cell,
                inputs=inputs_series,
                initial_state=hidden_state,
                sequence_length=self.batch_sizes)
        return states_series, hidden_size
    # End of hidden_layer()

    @debug()
    def input_layer(self):
        """
        @see RNNBase.input_layer
        """
        with tf.variable_scope(constants.INPUT):
            num_features = len(self.settings.rnn.input_names)
            self.batch_x_placeholder = tf.placeholder(
                dtype=tf.int32,
                shape=[self.settings.train.batch_size, self.settings.train.truncate, num_features],
                name='input_placeholder')
            unstacked_inputs = tf.unstack(self.batch_x_placeholder, axis=-1, name='unstack_inputs')
            input_vector_list = list()
            for index, inputs in enumerate(unstacked_inputs):
                input_vectors = token_to_vector(self.dataset.vocabulary_size[index], self.settings.rnn.hidden_size,
                                                inputs, self.settings.rnn.input_names[index])
                input_vector_list.append(input_vectors)
            inputs_series = tf.concat(input_vector_list, axis=-1, name='concatenate_inputs')
        return inputs_series
    # End of input_layer()
# End of MultiInputRNN
