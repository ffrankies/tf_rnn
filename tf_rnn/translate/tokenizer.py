"""Converts between tokens and indexes for tokenized input or output features.

@since 0.7.0
"""

from typing import List, Iterable, Dict, Any

import numpy as np

from . import Translator


class Tokenizer(Translator):
    """Converts indexes to tokens and tokens to indexes.
    """

    def __init__(self, index_to_token: List[Any], token_to_index: Dict[Any, int]) -> None:
        """Initializes the Indexer object.

        Params:
            index_to_token (list): Converts indexes into tokens
            token_to_index (dict or list): Converts tokens into indexes
        """
        self.index_to_token = index_to_token
        self.token_to_index = token_to_index
    # End of __init__()

    def to_human(self, index: int) -> Any:
        """Converts a single index into a token.

        Params:
            index (int): The index to convert into a token

        Returns:
            Any: The tokenized index
        """
        token = self.index_to_token[index]
        return token
    # End of to_human()

    def to_human_vector(self, index_vector: Iterable[int]) -> Iterable[Any]:
        """Converts a list of indexes into a list of tokens.

        Params:
            index_vector (Iterable[int]): The list of indexes to convert into tokens

        Returns:
            Iterable[Any]: The tokenized indexes
        """
        tokens = [self.to_human(index) for index in index_vector]
        return tokens
    # End of to_human_vector()

    def to_human_matrix(self, index_matrix: np.ndarray) -> np.ndarray:
        """Converts an array of indexes into an array of tokens.

        Params:
            index_matrix (np.ndarray[int]): The matrix of indexes to translate

        Returns:
            np.ndarray[Any]: The matrix of tokenized indexes
        """
        tokens = np.apply_along_axis(self.to_human_vector, 0, index_matrix)
        return tokens
    # End of to_human_matrix()

    def to_rnn(self, token: Any) -> int:
        """Converts a single token into an index.

        Params:
            token (Any): The token to convert into an index

        Returns:
            int: The token's index
        """
        index = self.token_to_index[token]
        return index
    # End of to_rnn()

    def to_rnn_vector(self, token_vector: Iterable[Any]) -> Iterable[int]:
        """Converts a list of token_vector into a list of indexes.

        Params:
            token_vector (Iterable[Any]): The vector of tokens to convert into indexes

        Returns:
            Iterable[int]: The vector of token indexes
        """
        indexes = [self.to_rnn(token) for token in token_vector]
        return indexes
    # End of to_rnn_vector()

    def to_rnn_matrix(self, token_matrix: np.ndarray) -> np.ndarray:
        """Converts an array of token_vector into an array of indexes.

        Params:
            token_matrix (np.ndarray[Any]): The matrix of tokens to be converted into their indexes

        Returns:
            np.ndarray[int]: The matrix of token indexes
        """
        indexes = np.apply_along_axis(self.to_rnn_vector, 0, token_matrix)
        return indexes
    # End of to_rnn_matrix()

    @classmethod
    def create(cls, data: List[List[Any]]) -> 'Tokenizer':
        """Creates a Tokenizer out of the given data. It is assumed that the data provided is a list of lists of
        object tokens.

        Params:
            data (List[List[Any]]): The token data out of which to create the tokenizer
        
        Returns:
            Tokenizer: The tokenizer created from the given data
        """
        flattened_data = np.ravel(data)
        token_set = set(flattened_data)
        index_to_token = list(token_set)
        token_to_index = dict((token, index) for index, token in enumerate(index_to_token))
        return Tokenizer(index_to_token, token_to_index)
    # End of create()
# End of Tokenizer()
