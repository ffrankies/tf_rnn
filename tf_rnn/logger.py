"""Wrapper for the python logging module - saves logs to multiple files based on severity level.

@since 0.6.4
"""

import logging
import logging.handlers
from typing import Callable, Any
from functools import wraps

from . import constants
from .utils import create_directory, Singleton


class Logger(object, metaclass=Singleton):
    """Wrapper for the python logging module. Contains multiple loggers with different severity levels. Each
    logger's output is logged to a separate file. Exposes the commonly used logging api: i.e.: one would interact
    with this logger by using the `error()`, `info()`, `debug()` and `trace()` methods.

    Params:
    - error_logger (logging.Logger): Logs information at the error severity level
    - info_logger (logging.Logger): Logs information at the info severity level
    - debug_logger (logging.Logger): Logs information at the debug severity level
    - trace_logger (logging.Logger): Logs information at the trace severity level
    """

    def __init__(self, log_directory: str = None):
        """Creates an instance of the Logger class.

        Stores logs in models/model_name/<Severity>.log files. The model name is obtained through the settings object,
        when the log directory is not given.

        Params:
        - log_directory (str): The path to the directory where the logs should be stored
        """
        if log_directory[-1] != '/':
            log_directory += '/'
            
        create_directory(log_directory)
        self.error_logger = self._create_logger(constants.ERROR, log_directory)
        self.info_logger = self._create_logger(constants.INFO, log_directory)
        self.debug_logger = self._create_logger(constants.DEBUG, log_directory)
        self.trace_logger = self._create_logger(constants.TRACE, log_directory)
    # End of __init__()

    def _create_logger(self, severity_level: str, log_directory: str) -> logging.Logger:
        """Creates a logger at the given level, using the severity level as the name of the logger

        Params:
        - severity_level (str): The severity level set for the logger
        - log_directory (str): The directory in which to save the log files
        """
        logger = logging.getLogger(severity_level)
        logger.setLevel(logging.INFO)  # Doesn't matter, so long as it logs to the same file
        
        # Logger will use up to 5 files for logging, 'rotating' the data between them as they get filled up.
        handler = logging.handlers.RotatingFileHandler(
            filename=log_directory+severity_level+'.log',
            maxBytes=1024*512,
            backupCount=5
        )

        # Set the format for the output messages
        formatter = logging.Formatter("%(asctime)s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    # End of _create_logger()

    def error(self, message: str):
        """Logs a message at the error severity level.

        Params:
        - message (str): The message to be logged
        """
        error_message = " %s:    %s" % (constants.ERROR, message)
        self.error_logger.info(error_message)
        self.info_logger.info(error_message)
        self.debug_logger.info(error_message)
        self.trace_logger.info(error_message)
    # End of error()

    def info(self, message: str):
        """Logs a message at the info severity level.

        Params:
        - message (str): The message to be logged
        """
        info_message = " %s :    %s" % (constants.INFO, message)
        self.info_logger.info(info_message)
        self.debug_logger.info(info_message)
        self.trace_logger.info(info_message)
    # End of info()

    def debug(self, message: str):
        """Logs a message at the debug severity level.

        Params:
        - message (str): The message to be logged
        """
        debug_message = " %s:    %s" % (constants.DEBUG, message)
        self.debug_logger.info(debug_message)
        self.trace_logger.info(debug_message)
    # End of debug()

    def trace(self, message: str):
        """Logs a message at the trace severity level.

        Params:
        - message (str): The message to be logged
        """
        trace_message = " %s:    %s" % (constants.TRACE, message)
        self.trace_logger.info(trace_message)
    # End of trace()
# End of Logger()


#
# Logging decorators that take a logger and a message as parameters
#
class LogDecorator(object):
    """Base class for a decorator for logging.
    """

    def __init__(self, message: str = None, logger: Logger = None):
        """Creates an instance of the log decorator.

        If no message is passed, logs the function name and the params given to the function.

        If no logger is passed, it expects the function to be a part of a class that contains a Logger object named
        logger.

        Params:
        - message (str): The message to be logged (default: None)
        - logger (Logger): The logger to be used (default: None)
        """
        self.message = message
        self.logger = logger
    # End of __init__()

    def _get_message(self, function: Callable) -> tuple:
        """Returns the message to log. If no message is passed to the decorator, logs the function called and its 
        arguments.

        Params:
        - function (Callable): The function that is being decorated

        Returns:
        - message (str): The message to log
        """
        if self.logger is None:
            self.logger = Logger()
        if self.message:
            message = self.message
        else:  # Use function name
            message = "{}".format(function.__name__)
        return message
    # End of _get_message()
# End of LogDecorator()


class error(LogDecorator):
    """Decorator for the call to logger.error()
    """

    def __call__(self, function: Callable) -> Callable:
        """Logs the given message at the error severity, and returns the function that the decorator wraps.

        Params:
        - function: The function wrapped by this decorator
        """
        @wraps(function)
        def wrapped_function(*args: tuple, **kwargs: dict) -> Any:
            """Returns the decorated function, which spews out the error log message before it is called.
            """
            message = self._get_message(function)
            self.logger.error(message)
            return function(*args, **kwargs)
        # End of wrapped_function

        return wrapped_function
    # End of __call__()
# End of error() decorator


class info(LogDecorator):
    """Decorator for the call to logger.info()
    """

    def __call__(self, function: Callable) -> Callable:
        """Logs the given message at the info severity, and returns the function that the decorator wraps.

        Params:
        - function: The function wrapped by this decorator
        """
        @wraps(function)
        def wrapped_function(*args: tuple, **kwargs: dict) -> Any:
            """Returns the decorated function, which spews out the info log message before it is called.
            """
            message = self._get_message(function)
            self.logger.info(message)
            return function(*args, **kwargs)
        # End of wrapped_function

        return wrapped_function
    # End of __call__()
# End of info() decorator


class debug(LogDecorator):
    """Decorator for the call to logger.debug()
    """

    def __call__(self, function: Callable) -> Callable:
        """Logs the given message at the debug severity, and returns the function that the decorator wraps.

        Params:
        - function: The function wrapped by this decorator
        """
        @wraps(function)
        def wrapped_function(*args: tuple, **kwargs: dict) -> Any:
            """Returns the decorated function, which spews out the debug log message before it is called.
            """
            message = self._get_message(function)
            self.logger.debug(message)
            return function(*args, **kwargs)
        # End of wrapped_function

        return wrapped_function
    # End of __call__()
# End of debug() decorator


class trace(LogDecorator):
    """Decorator for the call to logger.trace()
    """

    def __call__(self, function: Callable) -> Callable:
        """Logs the given message at the trace severity, and returns the function that the decorator wraps.

        Params:
        - function: The function wrapped by this decorator
        """
        @wraps(function)
        def wrapped_function(*args: tuple, **kwargs: dict) -> Any:
            """Returns the decorated function, which spews out the trace log message before it is called.
            """
            message = self._get_message(function)
            self.logger.trace(message)
            return function(*args, **kwargs)
        # End of wrapped_function

        return wrapped_function
    # End of __call__()
# End of trace() decorator
