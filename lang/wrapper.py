import time
import logging
from functools import wraps


def timethis(func):
    ''' Decorator that reports the execution time.

    Link:
        https://python3-cookbook.readthedocs.io/zh_CN/latest/c09/p01_put_wrapper_around_function.html

    Usage:
        @timethis
        def countdown(n):
            pass
    '''
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(func.__name__, end-start)
        return result
    return wrapper


def logged(level, name=None, message=None):
    """
    Add logging to a function. level is the logging
    level, name is the logger name, and message is the
    log message. If name and message aren't specified,
    they default to the function's module and name.

    Link:
        https://python3-cookbook.readthedocs.io/zh_CN/latest/c09/p04_define_decorator_that_takes_arguments.html

    Usage:
        # Example use
        @logged(logging.DEBUG)
        def add(x, y):
            return x + y
    """
    def decorate(func):
        logname = name if name else func.__module__
        log = logging.getLogger(logname)
        logmsg = message if message else func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            log.log(level, logmsg)
            return func(*args, **kwargs)
        return wrapper
    return decorate
