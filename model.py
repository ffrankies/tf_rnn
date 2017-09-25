"""
An RNN model implementation in tensorflow.

Copyright (c) 2017 Frank Derry Wanye

Date: 24 September, 2017
"""

import numpy as np
import tensorflow as tf
import logging
import ray
import time

from . import constants
from . import setup
from . import datasets
from . import saver
from . import tensorboard
from . import settings

class RNNModel(object):
    """
    A basic RNN implementation in tensorflow.
    """

    def __init__(self):
        """
        Constructor for an RNN Model.
        """
        self.settings = settings.Settings()
        self.model_path = saver.create_model_dir(self.settings.general.model_name)
        self.logger = setup.setup_logger(self.settings.logging, self.model_path)
        self.logger.info("RNN settings: %s" % self.settings)
        self.create_graph()
    # End of __init__()

    def create_graph(self):
        """
        Creates all internal tensorflow operations and variables inside a local graph and session.
        """
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.load_dataset()
            self._training()
            self.session = tf.Session(graph=self.graph)
            self._init_saver()
            self.session.run(tf.global_variables_initializer())
    # End of create_graph()

    def load_dataset(self):
        """
        Loads the dataset specified in the command-line arguments. Instantiates variables for the class.
        """
        dataset_params = datasets.load_dataset(self.logger, self.settings.rnn.dataset)
        self.data_type = dataset_params[0]
        self.token_level = dataset_params[1]
        # Skip vocabulary - we don't really need it
        self.index_to_token = dataset_params[3]
        self.token_to_index = dataset_params[4]
        self.vocabulary_size = len(self.index_to_token)
        # Don't need to keep the actual training data when creating batches.
        x_train = self.create_long_array(dataset_params[5])
        y_train = self.create_long_array(dataset_params[6])
        self.create_batches(x_train, y_train)
    # End of load_dataset()

    def create_long_array(self, matrix):
        """
        Concatenates multi-dimensional array into one long array for simpler training. Pads the resulting array until 
        it is divisible by the batch size.

        Params:
        matrix (numpy.ndarray): The multi-dimensional array to concatenate

        Return:
        numpy.array: The concatenated and padded array
        """
        array = np.array([])
        for row in matrix: array = np.append(array, row)
        while len(array) % self.settings.train.batch_size != 0: array = np.append(array, [array[-1]])
        return array
    # End of create_long_array()

    def create_batches(self, x_train, y_train):
        """
        Creates batches out of loaded data.

        Current implementation is very limited. It would probably be best to sort the training data based on length, 
        fill it up with placeholders so the sizes are standardized, and then break it up into batches.

        Params:
        x_train (numpy.array): The training examples
        y_train (numpy.array): The training labels
        """
        self.logger.info("Breaking input data into batches.")
        self.x_train_batches = x_train.reshape((self.settings.train.batch_size,-1))
        self.y_train_batches = y_train.reshape((self.settings.train.batch_size,-1))
        self.num_batches = len(self.x_train_batches)
        self.logger.info("Obtained %d batches." % self.num_batches)
    # End of create_batches() 

    def _training(self):
        """
        Creates tensorflow variables and operations needed for training.
        """
        logits_series = self._output_layer()
        with tf.variable_scope(constants.TRAINING):
            self.learning_rate = tf.Variable(
                initial_value=self.settings.train.learn_rate,
                dtype=tf.float32,
                name="learning_rate")
            outputs_series = tf.unstack(
                self.batch_y_placeholder, axis=1, 
                name="unstack_outputs_series")
            losses = []
            for logits, labels in zip(logits_series, outputs_series):
                labels = tf.to_int32(labels, "CastLabelsToInt")
                losses.append(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
            self.total_loss_fun = tf.reduce_mean(losses)
            self.train_step_fun = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.total_loss_fun)
            self.accuracy = self._calculate_accuracy(outputs_series)
    # End of _training()

    def _calculate_accuracy(self, labels):
        """
        Tensorflow operation that calculates the model's accuracy on a given minibatch.

        :type labels: Tensor
        :param labels: The output labels.

        :type return: Tensor
        :param return: The list of accuracies for each 'word' in the minibatch.
        """
        accuracy = []
        for predictions, labels in zip(self.predictions_series, labels):
            labels = tf.to_int64(labels, "CastLabelsToInt")
            predictions = tf.argmax(predictions, axis=1)
            accuracy.append(tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32)))
        return accuracy
    # End of _calculate_accuracy()

    def _output_layer(self):
        """
        Creates the tensorflow variables and operations needed to compute the network outputs.
        """
        states_series = self._hidden_layer()
        with tf.variable_scope(constants.OUTPUT):
            self.batch_y_placeholder = tf.placeholder(
                dtype=tf.float32,
                shape=np.shape(self.batch_x_placeholder),
                name="output_placeholder")
            self.out_weights = tf.Variable(
                initial_value=np.random.rand(self.settings.rnn.hidden_size, self.vocabulary_size), 
                dtype=tf.float32,
                name="out_weights")
            self.out_bias = tf.Variable(
                np.zeros((self.vocabulary_size)), 
                dtype=tf.float32,
                name="out_bias") 
            logits_series = [
                tf.nn.xw_plus_b(state, self.out_weights, self.out_bias, name="state_times_out_weights")
                for state in states_series] #Broadcasted addition
        with tf.variable_scope("predictions"):
            self.predictions_series = [tf.nn.softmax(logits) for logits in logits_series]
        return logits_series
    # End of _output_layer()

    def _hidden_layer(self):
        """
        Creates the tensorflow variables and operations needed to compute the hidden layer state.
        """
        inputs_series = self._input_layer()
        with tf.variable_scope(constants.HIDDEN):
            self.hidden_state_placeholder = tf.placeholder(
                dtype=tf.float32, 
                shape=[self.settings.train.batch_size, self.settings.rnn.hidden_size],
                name="hidden_state_placeholder")
            cell = tf.contrib.rnn.GRUCell(self.settings.rnn.hidden_size, reuse=tf.get_variable_scope().reuse)
            states_series, self.current_state = tf.contrib.rnn.static_rnn(
                cell=cell, 
                inputs=inputs_series, 
                initial_state=self.hidden_state_placeholder)
        return states_series
    # End of _hidden_layer()

    def _input_layer(self):
        """
        Creates the tensorflow variables and operations needed to perform the embedding lookup.
        """
        with tf.variable_scope(constants.EMBEDDING):
            self.batch_x_placeholder = tf.placeholder(
                dtype=tf.int32, 
                shape=[self.settings.train.batch_size, self.settings.train.truncate],
                name="input_placeholder")
            embeddings = tf.get_variable(
                name="word_embeddings",
                shape=[self.vocabulary_size, self.settings.rnn.hidden_size],
                dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(
                params=embeddings, ids=self.batch_x_placeholder,
                name="embedding_lookup")
            inputs_series = tf.unstack(
                inputs, axis=1, 
                name="unstack_inputs_series")
        return inputs_series
    # End of _input_layer()

    def _init_saver(self):
        """
        Creates the variables needed to save the model weights and tensorboard summaries.
        """
        self.run_dir = saver.load_meta(self.model_path)
        self.summary_writer, self.summary_ops = tensorboard.init_tensorboard(self)
        self.variables = ray.experimental.TensorFlowVariables(self.total_loss_fun, self.session)
    # End of _init_saver()

    def __pad_2d_matrix__(self, matrix, value=None):
        """
        Pads the rows of a 2d matrix with either a given value or the last value in each 
        row.

        :type matrix: nested list
        :param matrix: 2d matrix in python list form with variable row length.

        :type value: int
        :param value: the value to append to each row.

        :type return: nested list
        :param return: 2d matrix in python list form with a fixed row length.
        """
        self.logger.debug("Padding matrix with shape: ", matrix.shape)
        paddedMatrix = matrix
        maxRowLength = max([len(row) for row in paddedMatrix])
        for row in paddedMatrix:
            while len(row) < maxRowLength:
                row.append(value) if value is not None else row.append(row[-1])
        return paddedMatrix
    # End of __pad_2d_matrix__()

    def __list_to_numpy_array__(self, matrix):
        """
        Converts a list of list matrix to a numpy array of arrays.

        :type matrix: nested list
        :param matrix: 2d matrix in python list form.

        :type return: nested numpy array
        :param return: the matrix as a numpy array or arrays.
        """
        paddedMatrix = self.__pad_2d_matrix__(matrix, value=None)
        return np.array([np.array(row, dtype=int) for row in paddedMatrix], dtype=int)
    # End of __list_to_numpy_array__()