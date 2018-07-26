"""Holds the performance metrics for a model.

@since 0.6.1
"""

from . import Accumulator


class Metrics(object):
    """Holds the performance metrics for a given model.

    Instance Variables:
    - train (Accumulator): The accumulator that holds training performance metrics
    - valid (Accumulator): The accumulator that holds validation performance metrics
    - test (Accumulator): The accumulator that holds test performance metrics
    """

    def __init__(self, max_sequence_length: int):
        """Creates a metrics object with three empty accumulators.

        Params:
        - max_sequence_length (int): The maximum sequence length for this dataset
        """
        self.train = Accumulator(max_sequence_length)
        self.valid = Accumulator(max_sequence_length)
        self.test = Accumulator(max_sequence_length)
    # End of __init__()

    def advance(self):
        """Advances the training and validation accumulators to the next epoch.
        """
        self.train.next_epoch()
        self.valid.next_epoch()
    # End of advance()
# End of Metrics
