"""
Contains functions for setting up the input layer for a tensorflow-based RNN.

Copyright (c) 2017 Frank Derry Wanye

Date: 22 October, 2017
"""
import tensorflow as tf

def token_to_vector(vocabulary_size, hidden_size, batch_x_placeholder, dynamic_rnn=True):
    """
    Within a batch, converts tokens that represent classes into a vector that has the same size as the hidden layer.

    This step is equivalent to converting each token into a one-hot vector, multiplying that by a matrix
    of size (num_tokens, hidden_layer_size), and extracting the non-zero row from the result.

    Params:
    vocabulary_size (int): The number of known tokens
    hidden_size (int): The size of the hidden layer
    batch_x_placeholder (tf.placeholder): The placeholder for the input batch
    dynamic_rnn (bool): True if the RNN is a dynamic RNN

    Return:
    tensorflow.Variable: This inputs series that serve as the input to the hidden layer
    """
    embeddings = tf.get_variable(
        name="embedding_matrix",
        shape=[vocabulary_size, hidden_size],
        dtype=tf.float32)
    inputs_series = tf.nn.embedding_lookup(
        params=embeddings, ids=batch_x_placeholder,
        name="embedding_lookup")
    if not dynamic_rnn:
        inputs_series = tf.unstack(inputs, axis=1, name="unstack_inputs_series")
    return inputs_series
# End of word_inputs_series()