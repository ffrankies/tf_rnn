'''
Contains the Indexer object - it's used for converting tokens to indexes and indexes back to tokens.
@since 0.4.2
'''

from . import constants

class Indexer(object):
    '''
    Converts indexes to tokens and tokens to indexes based on the number of features in the dataset. It expects 
    every feature to have its own vocabulary, and thus its own index_to_token and token_to_index objects.

    Instance variables:
    - num_features (int): The number of features in the dataset
    - index_to_token (list): Converts indexes into tokens
    - token_to_index (dict or list): Converts tokens into indexes
    '''

    def __init__(self, num_features, index_to_token, token_to_index):
        '''
        Initializes the Indexer object.

        Params:
        - num_features (int): The number of features in the dataset
        - index_to_token (list): Converts indexes into tokens
        - token_to_index (dict or list): Converts tokens into indexes
        '''
        self.num_features = num_features
        self.index_to_token = index_to_token
        self.token_to_index = token_to_index
    # End of __init__()

    def to_token(self, index, feature_index=0):
        '''
        Converts a single index into a token.

        Params:
        - feature_index (int): The index of the feature to which the index belongs. Defaults to 0
        - index (int): The index to convert into a token
        '''
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
        '''
        Converts a list of indexes into a list of tokens.

        Params:
        - indexes (list): The lsit of indexes to convert into tokens
        - feature_index (int): The index of the feature to which the index belongs. Defaults to 0
        '''
        if feature_index >= self.num_features:
            raise ValueError('The feature_index was > number of features: %d > %d' % 
                (feature_index, self.num_features))
        if self.num_features == 1:
            tokens = [self.to_token(index) for index in indexes]
        else:
            tokens = [self.to_token(index, feature_index) for index in indexes]
        return tokens
    # End of to_tokens()

    def to_index(self, token, feature_index=0):
        '''
        Converts a single token into an index.

        Params:
        - index (str): The index to convert to a token
        - feature_index (int): The index of the feature to which the index belongs. Defaults to 0
        '''
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
        '''
        Converts a list of tokens into a list of indexes.

        Params:
        - tokens (list): The lsit of tokens to convert into indexes
        - feature_index (int): The index of the feature to which the index belongs. Defaults to 0
        '''
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