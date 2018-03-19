"""Contains the Indexer object - it's used for converting tokens to indexes and indexes back to tokens.

Copyright (c) 2017-2018 Frank Derry Wanye

@since 0.5.0
"""
import numpy as np


class Indexer(object):
    """Converts indexes to tokens and tokens to indexes based on the number of features in the dataset. It expects
    every feature to have its own vocabulary, and thus its own index_to_token and token_to_index objects.

    Instance variables:
    - num_features (int): The number of features in the dataset
    - index_to_token (list): Converts indexes into tokens
    - token_to_index (dict or list): Converts tokens into indexes
    """

    def __init__(self, num_features, index_to_token, token_to_index):
        """Initializes the Indexer object.

        Params:
        - num_features (int): The number of features in the dataset
        - index_to_token (list): Converts indexes into tokens
        - token_to_index (dict or list): Converts tokens into indexes
        """
        self.num_features = num_features
        self.index_to_token = index_to_token
        self.token_to_index = token_to_index
    # End of __init__()

    def to_token(self, index: int, feature_index: int = 0) -> str:
        """Converts a single index into a token.

        Params:
        - index (int): The index to convert into a token
        - feature_index (int): The index of the feature to which the index belongs. Defaults to 0

        Returns:
        - token (str): The tokenized index
        """
        if feature_index >= self.num_features:
            raise ValueError('The feature_index was > number of features: {} > {}'.format(
                feature_index, self.num_features))
        if self.num_features == 1:
            token = self.index_to_token[index]
        else:
            token = self.index_to_token[feature_index][index]
        return token
    # End of to_token()

    def to_tokens(self, indexes: list, feature_index: int = 0) -> list:
        """Converts a list of indexes into a list of tokens.

        Params:
        - indexes (list<int>): The list of indexes to convert into tokens
        - feature_index (int): The index of the feature to which the index belongs. Defaults to 0

        Returns:
        - tokens (list<str>): The tokenized indexes
        """
        if feature_index >= self.num_features:
            raise ValueError('The feature_index was > number of features: {} > {}'.format(
                feature_index, self.num_features))
        if self.num_features == 1:
            tokens = [self.to_token(index) for index in indexes]
        else:
            tokens = [self.to_token(index, feature_index) for index in indexes]
        return tokens
    # End of to_tokens()

    def to_token_array(self, indexes_array: np.array, features_present: list = None) -> np.array:
        """Converts an array of indexes into an array of tokens. If features_present is None, it is assumed that only
        the first index is present (features_present will be set to [0]).

        Params:
        - indexes_array (ndarray<int>): The array of indexes to convert into tokens
        - features_present (list<int>): The indexes of the features present in the array. Defaults to None

        Returns:
        - tokenized_array (np.ndarray<str>): The array of tokenized indexes
        """
        if not features_present:
            features_present = [0]
        indexes = np.array(indexes_array)
        if len(features_present) > 1:
            indexes = np.split(indexes, len(features_present), -1)
            indexes = [np.squeeze(index) for index in indexes]
            for feature_index in features_present:
                indexes[feature_index] = np.apply_along_axis(self.to_tokens, 0, indexes[feature_index], feature_index)
            indexes = np.stack(indexes, -1)
            indexes = np.squeeze(indexes)
        else:
            indexes = np.apply_along_axis(self.to_tokens, 0, indexes)
        return indexes
    # End of to_token_array()

    def to_index(self, token: str, feature_index: int = 0) -> int:
        """Converts a single token into an index.

        Params:
        - token (str): The token to convert into an index
        - feature_index (int): The index of the feature to which the index belongs. Defaults to 0

        Returns:
        - index (int): The indexed token
        """
        if feature_index >= self.num_features:
            raise ValueError('The feature_index was > number of features: {} > {}'.format(
                feature_index, self.num_features))
        if self.num_features == 1:
            index = self.token_to_index[token]
        else:
            index = self.token_to_index[feature_index][token]
        return index
    # End of to_index()

    def to_indexes(self, tokens: list, feature_index: int = 0) -> list:
        """Converts a list of tokens into a list of indexes.

        Params:
        - tokens (list<str>): The list of tokens to convert into indexes
        - feature_index (int): The index of the feature to which the index belongs. Defaults to 0

        Returns:
        - indexes (list<int>): The indexed tokens
        """
        if feature_index >= self.num_features:
            raise ValueError('The feature_index was > number of features: {} > {}'.format(
                feature_index, self.num_features))
        if self.num_features == 1:
            indexes = [self.to_token(token) for token in tokens]
        else:
            indexes = [self.to_token(token, feature_index) for token in tokens]
        return indexes
    # End of to_indexes()

    def to_index_array(self, tokens_array: np.array, features_present: list = None) -> np.array:
        """Converts an array of tokens into an array of indexes. If features_present is None, it is assumed that only
        the first index is present (features_present will be set to [0]).

        Params:
        - tokens_array (ndarray<int>): The array of tokens to convert into indexes
        - features_present (list<int>): The indexes of the features present in the array. Defaults to None

        Returns:
        - indexed_array (np.ndarray<str>): The array of indexed tokens
        """
        if not features_present:
            features_present = [0]
        tokens = np.array(tokens_array)
        if len(features_present) > 1:
            tokens = np.split(tokens, len(features_present), -1)
            tokens = [np.squeeze(token) for token in tokens]
            for feature_index in features_present:
                tokens[feature_index] = np.apply_along_axis(self.to_indexes, 0, tokens[feature_index], feature_index)
            indexed_array = np.stack(tokens, -1)
            indexed_array = np.squeeze(indexed_array)
        else:
            indexed_array = np.apply_along_axis(self.to_tokens, 0, indexed_array)
        return indexed_array
    # End of to_index_array()
# End of Indexer()
