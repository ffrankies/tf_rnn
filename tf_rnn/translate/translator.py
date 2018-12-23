"""Abstract base class for translating values between the RNN-readable format and the human-readable format.

@since 0.7.0
"""

import abc

from typing import Any, Iterable

import numpy as np


class Translator(object):
    """Translates RNN inputs and outputs between the RNN-readable format and the human-readable format.
    """

    @abc.abstractmethod
    def to_human(self, value: Any) -> Any:
        """Translates a single input or output value from the RNN-readable format to the human-readable format.

        Params:
            value (Any): The RNN-readable value to translate
        
        Returns:
            Any: The human-readable value
        """
    # End of to_human()

    @abc.abstractmethod
    def to_human_vector(self, value_vector: Iterable[Any]) -> Iterable[Any]:  
        """Translates a vector of input or output values from the RNN-readable format to the human-redable format.

        Params:
            value_vector (Iterable[Any]): The RNN-readable vector of values to translate

        Returns:
            Iterable[Any]: The human-readable vector of values
        """
    # End of to_human_vector()

    @abc.abstractmethod
    def to_human_matrix(self, value_matrix: np.ndarray) -> np.ndarray:
        """Translates a matrix of input or output values from the RNN-readable format to the human-readable format.

        Params:
            value_matrix (np.ndarray[Any]): The RNN-readable vector of values to translate

        Returns:
            np.ndarray[Any]: The human-readable vector of values
        """
    # End of to_human_matrix()

    @abc.abstractmethod
    def to_rnn(self, value: Any) -> Any:
        """Translates a single input or output value from the human-readable format to the RNN-readable format.

        Params:
            value (Any): The human-readable value to translate
        
        Returns:
            Any: The RNN-readable value
        """
    # End of to_rnn()

    @abc.abstractmethod
    def to_rnn_vector(self, value_vector: Iterable[Any]) -> Iterable[Any]:  
        """Translates a vector of input or output values from the human-readable format to the RNN-readable
        format.

        Params:
            value_vector (Iterable[Any]): The human-readable vector of values to translate

        Returns:
            Iterable[Any]: The RNN-readable vector of values
        """
    # End of to_rnn_vector()

    @abc.abstractmethod
    def to_rnn_matrix(self, value_matrix: np.ndarray) -> np.ndarray:
        """Translates a matrix of input or output values from the  human-readable format to the RNN-readable
        format.

        Params:
            value_matrix (np.ndarray[Any]): The human-readable vector of values to translate

        Returns:
            np.ndarray[Any]: The RNN-readable vector of values
        """
    # End of to_rnn_matrix()
# End of Translator()
