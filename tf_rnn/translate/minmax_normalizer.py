"""Uses min-max normalization to convert between normalized and original input or output features.

@since 0.7.0
"""

from typing import List, Iterable, Dict, Union

import numpy as np

from . import Translator


Numeric = Union[int, float]


class MinMaxNormalizer(Translator):
    """Uses min-max normalization to convert between normalized and original input or output features. The normalized
    values will always fall between -1 and 1, provided that the value provided is within [minimum, maximum].

    Min-max normalization works with the following function:
    normalized = (original - [(max + min) / 2]) / ([max - min] / 2)

    The original value can be obtained by changing the subject:
    original = ([normalized * (max - min)] + max + min) / 2
    """

    @classmethod
    def create(cls, data: np.ndarray) -> 'MinMaxNormalizer':
        """Creates a MinMaxNormalizer by finding the minimum and maximum of the given numeric data.

        Params:
            data (np.ndarray[Numeric]): The numeric data
        
        Returns:
            MinMaxNormalizer: The normalizer created from the numeric data
        """
        maximum = np.max(data)
        minimum = np.min(data)
        return cls(maximum, minimum)
    # End of create()

    def __init__(self, maximum: float, minimum: float) -> None:
        """Initializes the MinMaxNormalizer object.

        Params:
            maximum (float): The maximum original value of the feature in the dataset
            minimum (float): The minimum original value of the feature in the dataset
        """
        self.maximum = maximum
        self.minimum = minimum
    # End of __init__()

    def _normalize(self, original: Numeric) -> float:
        """Normalizes a single original value.

        Params:
            original (Numeric): The original, un-normalized value
        
        Returns:
            float: The normalized value
        """
        numerator = original - ((self.maximum + self.minimum) / 2)
        denominator= (self.maximum - self.minimum) / 2
        return numerator / denominator
    # End of _normalize()

    def _denormalize(self, normalized: float) -> float:
        """De-normalizes a normalized value.

        Params:
            normalized (float): The normalized value
        
        Returns:
            float: The original, de-normalized value
        """
        denormalized = (normalized * (self.maximum - self.minimum)) + self.maximum + self.minimum
        return denormalized / 2
    # End of _denormalize()

    def to_human(self, normalized: float) -> float:
        """Denormalized the given value.

        Params:
            normalized (float): The normalized value

        Returns:
            float: The denormalized value
        """
        denormalized = self._denormalize(normalized)
        return denormalized
    # End of to_human()

    def to_human_vector(self, normalized_vector: Iterable[float]) -> Iterable[float]:
        """Denormalizes the vector of normalized values.

        Params:
            normalized_vector (Iterable[float]): The vector of normalized values

        Returns:
            Iterable[float]: The denormalized vector of values
        """
        denormalized = [self.to_human(normalized) for normalized in normalized_vector]
        return denormalized
    # End of to_human_vector()

    def to_human_matrix(self, normalized_matrix: np.ndarray) -> np.ndarray:
        """Denormalizes the matrix of normalized values.

        Params:
            normalized_matrix (np.ndarray[float]): The matrix of normalized values

        Returns:
            np.ndarray[float]: The matrix of denormalized values
        """
        denormalized = np.apply_along_axis(self.to_human_vector, 0, normalized_matrix)
        return denormalized
    # End of to_human_matrix()

    def to_rnn(self, original: Numeric) -> float:
        """Normalizes an original, un-normalized value.

        Params:
            original (Numeric): The value to be normalized

        Returns:
            float: The normalized value
        """
        normalized = self._normalize(original)
        return normalized
    # End of to_rnn()

    def to_rnn_vector(self, original_vector: Iterable[Numeric]) -> Iterable[float]:
        """Normalizes a vector of original, un-normalized values.

        Params:
            original_vector (Iterable[float]): The vector of originals, un-normalized values

        Returns:
            Iterable[float]: The vector of normalized values
        """
        normalized = [self.to_rnn(original) for original in original_vector]
        return normalized
    # End of to_rnn_vector()

    def to_rnn_matrix(self, original_matrix: np.ndarray) -> np.ndarray:
        """Normalizes a matrix of original, un-normalized values.

        Params:
            original_matrix (np.ndarray[Numeric]): The matrix of original, un-normalized values
        Returns:
            np.ndarray[float]: The matrix of normalized values
        """
        normalized = np.apply_along_axis(self.to_rnn_vector, 0, original_matrix)
        return normalized
    # End of to_rnn_matrix()
# End of MinMaxNormalizer()
