"""Contains functions for setting up the input layer for a tensorflow-based RNN.

@since 0.6.1
"""

from typing import Tuple

import tensorflow as tf

from ..translate.translator import Translator
from ..translate.tokenizer import Tokenizer
from ..settings import RNNSettings


def format_feature(translator: Translator, settings: RNNSettings, feature_batch: tf.Tensor,
                   name: str) -> tf.Tensor:
    """Formats the feature for the RNN. Tokens get converted to vectors via embedding layers. Float features are not
    processed.

    Params:
        translator (Translator): The type of translator determines what will be done with the feature
        settings (RNNSettings): The RNN settings
        feature_batch (tf.Tensor): The unformatted feature values
        name (str): The name of the feature

    Returns:
        tf.Tensor: The formatted input feature
    """
    if isinstance(translator, Tokenizer):
        vocabulary_size = len(translator.index_to_token)
        embedding_size = settings.embed_size
        embedding_name = name + '_embedding'
        token_batch = tf.cast(feature_batch, tf.int32, name=name+'_to_int')
        return token_to_vector(vocabulary_size, embedding_size, token_batch, embedding_name)
    else:  # Feature is a float
        return feature_batch
# End of format_feature()


def token_to_vector(vocabulary_size: int, hidden_size: int, token_batch: tf.Tensor,
                    name: str = 'embedding') -> tf.Tensor:
    """Converts tokens that represent input classes into a vector that has the same size as the hidden layer.

    This step is equivalent to converting each token into a one-hot vector, multiplying that by a matrix
    of size (num_tokens, hidden_layer_size), and extracting the non-zero row from the result.

    Params:
        vocabulary_size (int): The size of the vocabulary
        hidden_size (int): The size of the hidden layer
        token_batch (tf.Tensor): The token batch to be converted to an embedding vector
        name (string): The name to be given to the embedding matrix

    Returns:
        tf.Tensor: This inputs series that serve as the input to the hidden layer
    """
    embeddings = tf.get_variable(name=name+'_matrix', shape=[vocabulary_size, hidden_size], dtype=tf.float32)
    inputs_series = tf.nn.embedding_lookup(params=embeddings, ids=token_batch, name=name+'_lookup')
    return inputs_series
# End of word_inputs_series()
