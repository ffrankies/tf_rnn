"""Contains functions for setting up the input layer for a tensorflow-based RNN.

@since 0.6.1
"""

import tensorflow as tf


def token_to_vector(vocabulary_size: int, hidden_size: int, token_batch: tf.Tensor,
                    name: str = 'embedding') -> tf.Tensor:
    """Converts tokens that represent input classes into a vector that has the same size as the hidden layer.

    This step is equivalent to converting each token into a one-hot vector, multiplying that by a matrix
    of size (num_tokens, hidden_layer_size), and extracting the non-zero row from the result.

    Params:
    - vocabulary_size (int): The size of the vocabulary
    - hidden_size (int): The size of the hidden layer
    - token_batch (tf.Tensor): The token batch to be converted to an embedding vector
    - name (string): The name to be given to the embedding matrix

    Returns:
    - inputs_series (tf.Tensor): This inputs series that serve as the input to the hidden layer
    """
    embeddings = tf.get_variable(name=name+'_matrix', shape=[vocabulary_size, hidden_size], dtype=tf.float32)
    inputs_series = tf.nn.embedding_lookup(params=embeddings, ids=token_batch, name=name+'_lookup')
    return inputs_series
# End of word_inputs_series()
