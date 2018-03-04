"""Holds different utility functions that don't entirely fit in any other module.

@since 0.5.0
"""

import os
from typing import Any


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


class Singleton(type):
    """Singleton metaclass. Objects that inherit this become Singleton objects. This implementation makes it so that
    subsequent calls to the Singleton class do not invoke the __init__() method.
    """

    _instances = {}  # Set of Singleton classes currently in use

    def __call__(cls: Any, *args: tuple, **kwargs: dict):
        """Whenever the Singleton class is declared, checks if an instance of that class has already been initiated.
        If it has, then return that instance. Otherwise, return a new instance.
        """
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
    # End of __call__()
# End of Singleton()
