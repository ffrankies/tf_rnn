"""Contains a list of input and output translators.

@since 0.7.0
"""

from typing import List, Any, Iterable, Type

import numpy as np

from .translator import Translator
from .minmax_normalizer import MinMaxNormalizer
from .tokenizer import Tokenizer
from .. import constants


class Translators(object):
    """Contains the input and output translators.
    """

    def __init__(self, output_indexes: List[int]) -> None:
        """Initializes a Translators object with an empty list of input and output translators.
        """
        self.num_input_translators = 0
        self.input_translators = list()  # type: List[Translator]
        self.num_output_translators = 0
        self.output_translators = list()  # type: List[Translator]
        self.output_indexes = output_indexes
    # End of __init__()

    def add_input_translator(self, translator: Translator):
        """Adds a translator to the list of input translators.

        Params:
            translator (Translator): The input translator to add
        """
        self.input_translators.append(translator)
        self.num_input_translators += 1
    # End of add_input_translator()

    def add_output_translator(self, translator: Translator):
        """Adds a translator to the list of output translators.

        Params:
            translator (Translator): The output translator to add
        """
        self.output_translators.append(translator)
        self.num_output_translators += 1
    # End of add_output_translator()

    def input_to_human(self, index: int, value: Any) -> Any:
        """Translates a single input value from the RNN-readable format to the human-readable format.

        Params:
            index (int): The index of the input translator (same as the index of the input feature)
            value (Any): The RNN-readable value to translate
        
        Returns:
            Any: The human-readable value
        """
        return self.input_translators[index].to_human(value)
    # End of input_to_human()

    def input_to_human_vector(self, index: int, value_vector: Iterable[Any]) -> Iterable[Any]:
        """Translates a vector of input values from the RNN-readable format to the human-redable format.

        Params:
            index (int): The index of the input translator (same as the index of the input feature)
            value_vector (Iterable[Any]): The RNN-readable vector of values to translate

        Returns:
            Iterable[Any]: The human-readable vector of values
        """
        return self.input_translators[index].to_human_vector(value_vector)
    # End of input_to_human_vector()

    def input_to_human_matrix(self, index: int, value_matrix: np.ndarray) -> np.ndarray:
        """Translates a matrix of input values from the RNN-readable format to the human-readable format.

        Params:
            index (int): The index of the input translator (same as the index of the input feature)
            value_matrix (np.ndarray[Any]): The RNN-readable vector of values to translate

        Returns:
            np.ndarray[Any]: The human-readable vector of values
        """
        return self.input_translators[index].to_human_matrix(value_matrix)
    # End of input_to_human_matrix()

    def input_to_rnn(self, index: int, value: Any) -> Any:
        """Translates a single input value from the human-readable format to the RNN-readable format.

        Params:
            index (int): The index of the input translator (same as the index of the input feature)
            value (Any): The human-readable value to translate
        
        Returns:
            Any: The RNN-readable value
        """
        return self.input_translators[index].to_rnn(value)
    # End of input_to_rnn()

    def input_to_rnn_vector(self, index: int, value_vector: Iterable[Any]) -> Iterable[Any]:  
        """Translates a vector of input values from the human-readable format to the RNN-readable
        format.

        Params:
            index (int): The index of the input translator (same as the index of the input feature)
            value_vector (Iterable[Any]): The human-readable vector of values to translate

        Returns:
            Iterable[Any]: The RNN-readable vector of values
        """
        return self.input_translators[index].to_rnn_vector(value_vector)
    # End of input_to_rnn_vector()

    def input_to_rnn_matrix(self, index: int, value_matrix: np.ndarray) -> np.ndarray:
        """Translates a matrix of input values from the  human-readable format to the RNN-readable
        format.

        Params:
            index (int): The index of the input translator (same as the index of the input feature)
            value_matrix (np.ndarray[Any]): The human-readable vector of values to translate

        Returns:
            np.ndarray[Any]: The RNN-readable vector of values
        """
        translator = self.input_translators[index]
        return translator.to_rnn_matrix(value_matrix)
    # End of input_to_rnn_matrix()

    def output_to_human(self, index: int, value: Any) -> Any:
        """Translates a single output value from the RNN-readable format to the human-readable format.

        Params:
            index (int): The index of the output translator (same as the index of the output feature)
            value (Any): The RNN-readable value to translate
        
        Returns:
            Any: The human-readable value
        """
        return self.output_translators[index].to_human(value)
    # End of output_to_human()

    def output_to_human_vector(self, index: int, value_vector: Iterable[Any]) -> Iterable[Any]:  
        """Translates a vector of output values from the RNN-readable format to the human-redable format.

        Params:
            index (int): The index of the output translator (same as the index of the output feature)
            value_vector (Iterable[Any]): The RNN-readable vector of values to translate

        Returns:
            Iterable[Any]: The human-readable vector of values
        """
        return self.output_translators[index].to_human_vector(value_vector)
    # End of output_to_human_vector()

    def output_to_human_matrix(self, index: int, value_matrix: np.ndarray) -> np.ndarray:
        """Translates a matrix of output values from the RNN-readable format to the human-readable format.

        Params:
            index (int): The index of the output translator (same as the index of the output feature)
            value_matrix (np.ndarray[Any]): The RNN-readable vector of values to translate

        Returns:
            np.ndarray[Any]: The human-readable vector of values
        """
        return self.output_translators[index].to_human_matrix(value_matrix)
    # End of output_to_human_matrix()

    def output_to_rnn(self, index: int, value: Any) -> Any:
        """Translates a single output value from the human-readable format to the RNN-readable format.

        Params:
            index (int): The index of the output translator (same as the index of the output feature)
            value (Any): The human-readable value to translate
        
        Returns:
            Any: The RNN-readable value
        """
        return self.output_translators[index].to_rnn(value)
    # End of output_to_rnn()

    def output_to_rnn_vector(self, index: int, value_vector: Iterable[Any]) -> Iterable[Any]:  
        """Translates a vector of output values from the human-readable format to the RNN-readable
        format.

        Params:
            index (int): The index of the output translator (same as the index of the output feature)
            value_vector (Iterable[Any]): The human-readable vector of values to translate

        Returns:
            Iterable[Any]: The RNN-readable vector of values
        """
        return self.output_translators[index].to_rnn_vector(value_vector)
    # End of output_to_rnn_vector()

    def output_to_rnn_matrix(self, index: int, value_matrix: np.ndarray) -> np.ndarray:
        """Translates a matrix of output values from the  human-readable format to the RNN-readable
        format.

        Params:
            index (int): The index of the output translator (same as the index of the output feature)
            value_matrix (np.ndarray[Any]): The human-readable vector of values to translate

        Returns:
            np.ndarray[Any]: The RNN-readable vector of values
        """
        return self.output_translators[index].to_rnn_matrix(value_matrix)
    # End of output_to_rnn_matrix()

    def translate_all(self, sequence_data: np.ndarray) -> np.ndarray:
        """Translates a matrix of all features to their RNN-readable values.

        Params:
            sequence_data (np.ndarray[Any]): The input features dataset
        
        Returns:
            np.ndarray[Any]: The RNN-readable versions of the input features
        """
        print("=====First sequence to be translated (shape = {} dtype = {})=====\n{}".format(
            np.shape(sequence_data), sequence_data.dtype, sequence_data[0]))
        expected_dtype = self._expected_rnn_input_dtype(sequence_data.dtype.names)
        stacked = np.empty(sequence_data.shape, dtype=expected_dtype)
        for feature_index, feature_name in enumerate(sequence_data.dtype.names):
            feature_data = sequence_data[feature_name]
            print("dtype of feature {} = {}\n{}".format(feature_index, feature_data.dtype, feature_data[0]))
            translated = self.input_to_rnn_matrix(feature_index, feature_data)
            stacked[feature_name] = translated
        print("=====Stacked Sequence (shape = {} dtype = {})=====\n{}".format(
            stacked.shape, stacked.dtype, stacked[0]
        ))
        return stacked
    # End of translate_all()

    def _expected_rnn_input_dtype(self, feature_names: List[str]) -> np.dtype:
        """
        """
        types = list()  # type: List[Type]
        for translator in self.input_translators:
            if isinstance(translator, MinMaxNormalizer):
                types.append(float)
            else:
                types.append(int)
        expected_dtype_desc = list(zip(feature_names, types))
        return np.dtype(expected_dtype_desc)
    # End of _expected_types()

    def output_size(self) -> int:
        """Calculates how many array elements are needed to encode the outputs of all the output translators.
        
        Returns:
            int: The number of array elements needed to encode the output of all the output translators
        """
        out_size = 0
        for translator in self.output_translators:
            out_size += translator.output_size()
        return out_size
    # End of output_size()
    
    @classmethod
    def create(cls, output_indexes: List[int], translator_types: List[str], sequence_data: np.ndarray) -> 'Translators':
        """Creates the necessary translators from the given data.

        Assumes that the correct translator_types have been provided by the user.
        Casts all input features translated by a MinMaxNormalizer to floats.
        
        Params:
            output_indexes (List[int]): The feature indexes that represent output data (data to be predicted)
            translator_types (List[str]): The list of types of translators to create
            sequence_data (np.ndarray): The data from which to create the translators
        
        Returns:
            Translators: The translators created from the data
        """
        print('=====Creating Translators=====')
        translators = Translators(output_indexes)
        if len(np.shape(sequence_data)) == 2:  # Only one input/output
            sequence_data = sequence_data.reshape(np.shape(sequence_data) + (1,))
        for feature_index, feature_name in enumerate(sequence_data.dtype.names):
            feature_data = sequence_data[feature_name]
            translator_type = translator_types[feature_index]
            if translator_type == constants.TRANSLATOR_TOKENIZER:
                translator = Tokenizer.create(feature_data)  # type: Translator
            elif translator_type == constants.TRANSLATOR_MINMAX:
                translator = MinMaxNormalizer.create(feature_data)
            else:
                raise ValueError("Translator type {} is not a known translator type".format(translator_type))
            translators.add_input_translator(translator)
            if feature_index in output_indexes:
                translators.add_output_translator(translator)
        return translators
    # End of create()
# End of Translators()
