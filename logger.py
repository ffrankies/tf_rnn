"""
Wrapper for the python logging module - saves logs to multiple files based on severity level.

@since 0.5.0
"""

import logging
import logging.handlers

from . import constants
from . import utils


class Logger(object):
    """Wrapper for the python logging module. Contains multiple loggers with different severity levels. Each
    logger's output is logged to a separate file. Exposes the commonly used logging api: i.e.: one would interact
    with this logger by using the `error()`, `info()`, `debug()` and `trace()` methods.

    Params:
    - error_logger (logging.Logger): Logs information at the error severity level
    - info_logger (logging.Logger): Logs information at the info severity level
    - debug_logger (logging.Logger): Logs information at the debug severity level
    - trace_logger (logging.Logger): Logs information at the trace severity level
    """

    def __init__(self, log_directory):
        """Creates an instance of the Logger class.

        Params:
        - log_directory (str): The name of the directory in which to save the log files
        """
        utils.create_directory(log_directory)
        self.error_logger = self.create_logger(constants.ERROR, log_directory)
        self.info_logger = self.create_logger(constants.INFO, log_directory)
        self.debug_logger = self.create_logger(constants.DEBUG, log_directory)
        self.trace_logger = self.create_logger(constants.TRACE, log_directory)
    # End of __init__()

    def create_logger(self, severity_level, log_directory):
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
    # End of create_logger()

    def error(self, message):
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

    def info(self, message):
        """Logs a message at the info severity level.

        Params:
        - message (str): The message to be logged
        """
        info_message = " %s :    %s" % (constants.INFO, message)
        self.info_logger.info(info_message)
        self.debug_logger.info(info_message)
        self.trace_logger.info(info_message)
    # End of info()

    def debug(self, message):
        """Logs a message at the debug severity level.

        Params:
        - message (str): The message to be logged
        """
        debug_message = " %s:    %s" % (constants.DEBUG, message)
        self.debug_logger.info(debug_message)
        self.trace_logger.info(debug_message)
    # End of debug()

    def trace(self, message):
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
class error(object):
    """Decorator for the call to logger.error()
    """

    def __init__(self, message):
        """Creates an instance of the error decorator.

        Params:
        - message (str): The message to log
        """
        self.message = message
    # End of __init__()

    def __call__(self, function):
        """Logs the given message at the error severity, and returns the function that the decorator wraps.

        Params:
        - function: The function wrapped by this decorator
        """
        message = self.message

        def wrapped_function(self, *args, **kwargs):
            """Returns the decorated function, which spews out the error log message before it is called.
            """
            self.logger.error(message)
            return function(self, *args, **kwargs)
        # End of wrapped_function

        return wrapped_function
    # End of __call__()
# End of error() decorator


class info(object):
    """Decorator for the call to logger.info()
    """

    def __init__(self, message):
        """Creates an instance of the info decorator.

        Params:
        - message (str): The message to log
        """
        self.message = message
    # End of __init__()

    def __call__(self, function):
        """Logs the given message at the info severity, and returns the function that the decorator wraps.

        Params:
        - function: The function wrapped by this decorator
        """
        message = self.message

        def wrapped_function(self, *args, **kwargs):
            """Returns the decorated function, which spews out the info log message before it is called.
            """
            self.logger.info(message)
            return function(self, *args, **kwargs)
        # End of wrapped_function

        return wrapped_function
    # End of __call__()
# End of info() decorator


class debug(object):
    """Decorator for the call to logger.debug()
    """

    def __init__(self, message):
        """Creates an instance of the debug decorator.

        Params:
        - message (str): The message to log
        """
        self.message = message
    # End of __init__()

    def __call__(self, function):
        """Logs the given message at the debug severity, and returns the function that the decorator wraps.

        Params:
        - function: The function wrapped by this decorator
        """
        message = self.message

        def wrapped_function(self, *args, **kwargs):
            """Returns the decorated function, which spews out the debug log message before it is called.
            """
            self.logger.debug("{} (args: {!s:.100} kwargs: {!s:.100})".format(message, args, kwargs))
            return function(self, *args, **kwargs)
        # End of wrapped_function

        return wrapped_function
    # End of __call__()
# End of debug() decorator


class trace(object):
    """Decorator for the call to logger.trace()
    """

    def __init__(self, message):
        """Creates an instance of the trace decorator.

        Params:
        - message (str): The message to log
        """
        self.message = message
    # End of __init__()

    def __call__(self, function):
        """Logs the given message at the trace severity, and returns the function that the decorator wraps.

        Params:
        - function: The function wrapped by this decorator
        """
        message = self.message

        def wrapped_function(self, *args, **kwargs):
            """Returns the decorated function, which spews out the trace log message before it is called.
            """
            self.logger.trace("{} (args: {!s:.100} kwargs: {!s:.100})".format(message, args, kwargs))
            return function(self, *args, **kwargs)
        # End of wrapped_function

        return wrapped_function
    # End of __call__()
# End of trace() decorator
