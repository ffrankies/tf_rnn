"""Contains functions for setting up the hidden layer for a tensorflow-based RNN.

@since 0.6.1
"""

import tensorflow as tf


def layered_state_tuple(num_layers: int, batch_size: int, hidden_size: int) -> tuple:
    """Constructs a tuple from the hidden state placeholder.

    Params:
    - num_layers (int): The number of layers in the RNN
    - batch_size (int): The batch size of the RNN model
    - hidden_size (int): The size of the hidden layers in the RNN model

    Returns:
    - hidden_state (tuple): The current hidden state to be passed into the dynamic_rnn
    - hidden_state_placeholder (tf.placeholder): The placeholder for the hidden state
    - hidden_state_shape (list): The shape of the hidden state
    """
    hidden_state_shape = [num_layers, batch_size, hidden_size]
    hidden_state_placeholder = tf.placeholder(
        dtype=tf.float32,
        shape=[num_layers, batch_size, hidden_size],
        name="hidden_state_placeholder")
    unpacked_hidden_state = tf.unstack(hidden_state_placeholder, axis=0, name="unpack_hidden_state")
    hidden_state = tuple(unpacked_hidden_state)
    return hidden_state, hidden_state_placeholder, hidden_state_shape
# End of layered_state_tuple()


def rnn_cell(num_layers: int, hidden_size: int, dropout: float) -> tf.contrib.rnn.MultiRNNCell:
    """
    Creates a multi-layered RNN cell with dropout.

    Params:
    - num_layers (int): The number of layers in the RNN
    - hidden_size (int): The size of the hidden layers in the RNN model
    - dropout (int): The fraction of output to keep during dropout

    Returns:
    - cell (MultiRNNCell): The cell of the given RNN
    """
    cell = tf.contrib.rnn.GRUCell(hidden_size)
    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout)
    cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers)
    return cell
# End of rnn_cell()
