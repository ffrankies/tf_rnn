"""
An RNN model implementation in tensorflow.

Copyright (c) 2017 Frank Derry Wanye

Date: 12 October, 2017
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
from . import batchmaker

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
            self.training()
            self.session = tf.Session(graph=self.graph)
            self.init_saver()
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
        # x_train = self.create_long_array(dataset_params[5])
        # y_train = self.create_long_array(dataset_params[6])
        self.create_batches(dataset_params[5], dataset_params[6])
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
        end_token = self.token_to_index[constants.END_TOKEN]
        self.inputs, self.labels, self.sizes = batchmaker.make_batches(x_train, y_train,
            self.settings.train.batch_size, self.settings.train.truncate, end_token)
        # self.x_train_batches = x_train.reshape((self.settings.train.batch_size,-1))
        # self.y_train_batches = y_train.reshape((self.settings.train.batch_size,-1))
        self.num_batches = len(self.inputs)
        self.logger.info("Obtained %d batches." % self.num_batches)
    # End of create_batches()

    def training(self):
        """
        Creates tensorflow variables and operations needed for training.
        """
        total_loss = self.performance_evaluation()
        with tf.variable_scope(constants.TRAINING):
            self.learning_rate = tf.Variable(
                initial_value=self.settings.train.learn_rate,
                dtype=tf.float32,
                name="learning_rate")
            self.train_step_fun = tf.train.AdagradOptimizer(self.learning_rate).minimize(total_loss)
    # End of training()

    def performance_evaluation(self):
        """
        Evaluates the performance of the network on a given minibatch.
        """
        logits_series = self.output_layer()
        with tf.variable_scope(constants.PERFORMANCE):
            row_lengths_series = tf.unstack(self.batch_sizes, name="unstack_batch_sizes")
            labels_series = tf.unstack(self.batch_y_placeholder, axis=1, name="unstack_labels_series")
            self.accuracy = self.calculate_accuracy(labels_series)
            self.total_loss_op = self.calculate_loss(logits_series, labels_series, row_lengths_series)
        return self.total_loss_op
    # End of calculate_loss()

    def calculate_accuracy(self, labels_series):
        """
        Tensorflow operation that calculates the model's accuracy on a given minibatch.

        Params:
        labels_series (tf.Tensor): True labels for each input

        Return:
        tf.Tensor: The average accuracy for each row in the minibatch
        """
        with tf.variable_scope(constants.ACCURACY):
            accuracy = []
            for predictions, labels in zip(self.predictions_series, labels_series):
                labels = tf.to_int64(labels, "CastLabelsToInt")
                predictions = tf.argmax(predictions, axis=1)
                accuracy.append(tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32)))
        return accuracy
    # End of calculate_accuracy()

    def calculate_loss(self, logits_series, labels_series, row_lengths_series):
        """
        Calculates the loss at a given minibatch.

        Params:
        logits_series (tf.Tensor): Calculated probabilities for each class for each input after training
        labels_series (tf.Tensor): True labels for each input
        row_lengths_series (tf.Tensor): The true, un-padded lengths of each row in the minibatch

        Return:
        tf.Tensor: The calculated average loss for this minibatch
        """
        with tf.variable_scope(constants.LOSS_CALC):
            loss_sum = 0.0
            num_valid_rows = 0.0
            for logits, labels, row_length in zip(logits_series, labels_series, row_lengths_series):
                # row_length = tf.to_int32(row_length, name="CastRowLengthToInt")
                ans = tf.greater(row_length, 0)
                num_valid_rows = tf.cond(ans, lambda: num_valid_rows + 1, lambda: num_valid_rows + 0)
                logits = logits[:row_length, :]
                labels = tf.to_int32(labels[:row_length], "CastLabelsToInt")
                row_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
                mean_loss = tf.cond(ans, lambda: tf.reduce_mean(row_losses[:row_length]), lambda: 0.0)
                loss_sum += mean_loss
            total_loss_op = loss_sum / num_valid_rows # Can't use reduce_mean because there will be 0s there
        return total_loss_op
    # End of calculate_loss()

    def output_layer(self):
        """
        Creates the tensorflow variables and operations needed to compute the network outputs.
        """
        states_series = self.hidden_layer()
        with tf.variable_scope(constants.OUTPUT):
            states_series = tf.unstack(states_series, axis=1, name="unstack_states_series")
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
    # End of output_layer()

    def hidden_layer(self):
        """
        Creates the tensorflow variables and operations needed to compute the hidden layer state.
        """
        inputs_series = self.input_layer()
        with tf.variable_scope(constants.HIDDEN):
            self.batch_sizes = tf.placeholder(
                dtype=tf.int32,
                shape=[self.settings.train.batch_size],
                name="batch_sizes")
            self.hidden_state_placeholder = tf.placeholder(
                dtype=tf.float32,
                shape=[self.settings.train.batch_size, self.settings.rnn.hidden_size],
                name="hidden_state_placeholder")
            cell = tf.contrib.rnn.GRUCell(self.settings.rnn.hidden_size)
            states_series, self.current_state = tf.nn.dynamic_rnn(
                cell=cell,
                inputs=inputs_series,
                initial_state=self.hidden_state_placeholder,
                sequence_length=self.batch_sizes)
        return states_series
    # End of hidden_layer()

    def input_layer(self):
        """
        Creates the tensorflow variables and operations needed to perform the embedding lookup.
        """
        with tf.variable_scope(constants.INPUT):
            self.batch_x_placeholder = tf.placeholder(
                dtype=tf.int32,
                shape=[self.settings.train.batch_size, self.settings.train.truncate],
                name="input_placeholder")
            if self.data_type == constants.TYPE_CHOICES[0]: # data type = 'text'
                inputs_series = self.token_to_vector()
            else:
                print("ERROR: Numeric inputs cannot be handled yet.")
                exit(-1)
        return inputs_series
    # End of input_layer()

    def token_to_vector(self):
        """
        Within a batch, converts tokens that represent classes into a vector that has the same size as the hidden layer.

        This step is equivalent to converting each token into a one-hot vector, multiplying that by a matrix
        of size (num_tokens, hidden_layer_size), and extracting the non-zero row from the result.

        Return:
        tensorflow.Variable: This inputs series that serve as the input to the hidden layer
        """
        embeddings = tf.get_variable(
            name="embedding_matrix",
            shape=[self.vocabulary_size, self.settings.rnn.hidden_size],
            dtype=tf.float32)
        inputs_series = tf.nn.embedding_lookup(
            params=embeddings, ids=self.batch_x_placeholder,
            name="embedding_lookup")
        # inputs_series = tf.unstack(
        #     inputs, axis=1,
        #     name="unstack_inputs_series")
        return inputs_series
    # End of word_inputs_series()

    def init_saver(self):
        """
        Creates the variables needed to save the model weights and tensorboard summaries.
        """
        self.run_dir = saver.load_meta(self.model_path)
        self.summary_writer, self.summary_ops = tensorboard.init_tensorboard(self)
        self.variables = ray.experimental.TensorFlowVariables(self.total_loss_op, self.session)
    # End of init_saver()

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
