"""Holds different utility functions that don't entirely fit in any other module.

@since 0.5.0
"""

import os


def create_directory(directory):
    """Creates a directory if it does not exist. 

    Params:
    - directory (str): The path to the directory to be created

    Raises:
    - OSError: if the directory creation fails
    """
    try:
        if os.path.dirname(directory):
            os.makedirs(os.path.dirname(directory), exist_ok=True)  # Python 3.2+
    except TypeError:
        try:  # Python 3.2-
            os.makedirs(directory)
        except OSError as exception:
            if exception.errno != 17:
                raise
# End of create_directory()