import functools
import time

__all__ = ["Timer", "log", "time_string", "tic_toc", "is_notebook"]


# -------------------------------------- Various Timer  --------------------------------------#
class Timer(object):
    """ A timer 
    """

    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.t_start = time.time()

    def __exit__(self, type, value, traceback):
        print("==> [%s]:\t" % self.name, end="")
        self.time_elapsed = time.time() - self.t_start
        print("Elapsed Time: %s (s)" % self.time_elapsed)


def tic_toc(func):
    """Print the time consumption of a certain function when a it is excuted.
    Args:
        func: An function.
    Usage:
        @tic_toc
        def func():
            pass
    """
    @functools.wraps(func)
    def wrapper(*args, **kw):
        tic = time.time()
        res = func(*args, **kw)
        print("==> [%s] executed in %.4f s." %
              (func.__name__, time.time() - tic))
        return res
    return wrapper


# -------------------------------------- Basic Functions  --------------------------------------#
def log(*snippets, end=None, tag="INFO", prefix=""):
    """ Easily replace print function to get a log-like format output with time and tag.
    """
    print(f"{prefix}[{tag}]", time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()) + " " + "".join([str(s) for s in snippets]),
          end=end)


def time_string():
    """ Generate a time string from year to second.
    """
    return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())


def is_notebook():
    """ Retrun a boolean value showing whether the code now is running on jupyter notebook.
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter
