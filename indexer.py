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

    def to_token(self, index, feature_index=0):
        """Converts a single index into a token.

        Params:
        - feature_index (int): The index of the feature to which the index belongs. Defaults to 0
        - index (int): The index to convert into a token
        """
        if feature_index >= self.num_features:
            raise ValueError('The feature_index was > number of features: %d > %d' %
                             (feature_index, self.num_features))
        if self.num_features == 1:
            token = self.index_to_token[index]
        else:
            token = self.index_to_token[feature_index][index]
        return token
    # End of to_token()

    def to_tokens(self, indexes, feature_index=0):
        """Converts a list of indexes into a list of tokens.

        Params:
        - indexes (list): The lsit of indexes to convert into tokens
        - feature_index (int): The index of the feature to which the index belongs. Defaults to 0
        """
        # print('Tokenizing {} with feature_index = {}'.format(indexes, feature_index))
        if feature_index >= self.num_features:
            raise ValueError('The feature_index was > number of features: %d > %d' %
                             (feature_index, self.num_features))
        if self.num_features == 1:
            tokens = [self.to_token(index) for index in indexes]
        else:
            tokens = [self.to_token(index, feature_index) for index in indexes]
        return tokens
    # End of to_tokens()

    def to_tokens_array(self, indexes_array, features_present=[0]):
        """Converts an array of indexes into an array of tokens.

        Params:
        - indexes_array (ndarray<int>): The lsit of indexes to convert into tokens
        - features_present (list<int>): The indexes of the features present in the array. Defaults to [0]
        """
        indexes = np.array(indexes_array)
        if len(features_present) > 1:
            print('Indexes before split: ', indexes[0])
            indexes = np.split(indexes, len(features_present), -1)
            indexes = [np.squeeze(index) for index in indexes]
            print('Try: ', self.index_to_token[2][8])
            print('Split indexes[0]: ', indexes[0][0])
            print('Split indexes[1]: ', indexes[1][0])
            print('Split indexes[2]: ', indexes[2][0])
            print('Num split indexes: ', len(indexes))
            for feature_index in features_present:
                indexes[feature_index] = np.apply_along_axis(self.to_tokens, 0, indexes[feature_index], feature_index)
            indexes = np.stack(indexes, -1)
            indexes = np.squeeze(indexes)
            print('Indexes after tokenization: ', indexes[3])
        else:
            indexes = np.apply_along_axis(self.to_tokens, 0, indexes)
        return indexes
    # End of to_tokens_array()

    def to_index(self, token, feature_index=0):
        """Converts a single token into an index.

        Params:
        - index (str): The index to convert to a token
        - feature_index (int): The index of the feature to which the index belongs. Defaults to 0
        """
        if feature_index >= self.num_features:
            raise ValueError('The feature_index was > number of features: %d > %d' %
                             (feature_index, self.num_features))
        if self.num_features == 1:
            index = self.token_to_index[token]
        else:
            index = self.token_to_index[feature_index][token]
        return index
    # End of to_index()

    def to_indexes(self, tokens, feature_index=0):
        """Converts a list of tokens into a list of indexes.

        Params:
        - tokens (list): The lsit of tokens to convert into indexes
        - feature_index (int): The index of the feature to which the index belongs. Defaults to 0
        """
        if feature_index >= self.num_features:
            raise ValueError('The feature_index was > number of features: %d > %d' %
                             (feature_index, self.num_features))
        if self.num_features == 1:
            indexes = [self.to_token(token) for token in tokens]
        else:
            indexes = [self.to_token(token, feature_index) for token in tokens]
        return indexes
    # End of to_indexes()
# End of Indexer()
