import logging
import sys


def setLogger(logger, logLevel='info'):
    """ Function to set up the handle error logging.
    logger (obj) = a logger object

    logLevel (str) = level of information to print out, options are {info, debug} [Default: info]

    """
    # Determine log level
    if logLevel == 'info':
        _level = logging.INFO
    elif logLevel == 'debug':
        _level = logging.DEBUG

    # Set the level in logger
    logger.setLevel(_level)

    # Set the log format
    logfmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Set logger output to STDOUT and STDERR
    loghandler = logging.StreamHandler(stream=sys.stdout)
    errhandler = logging.StreamHandler(stream=sys.stderr)

    # Set logging level for the different output handlers.
    # ERRORs to STDERR, and everything else to STDOUT
    loghandler.setLevel(_level)
    errhandler.setLevel(logging.ERROR)

    # Format the log handlers
    loghandler.setFormatter(logfmt)
    errhandler.setFormatter(logfmt)

    # Add handler to the main logger
    logger.addHandler(loghandler)
    logger.addHandler(errhandler)
