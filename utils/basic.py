import functools
import time


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.t_start = time.time()

    def __exit__(self, type, value, traceback):
        print("==> [%s]:\t" % self.name, end="")
        self.time_elapsed = time.time() - self.t_start
        print("Elapsed Time: %s (s)" % self.time_elapsed)


def log(*snippets, end=None, tag="INFO", prefix=""):
    print(f"{prefix}[{tag}]", time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()) + " " + "".join([str(s) for s in snippets]),
          end=end)

def time_string():
    return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())


def tic_toc(func):
    """Print the time consumption of a certain function when a it is excuted.
    Args:
        func: An function.
    """
    @functools.wraps(func)
    def wrapper(*args, **kw):
        tic = time.time()
        res = func(*args, **kw)
        print("==> [%s] executed in %.4f s." %
              (func.__name__, time.time() - tic))
        return res
    return wrapper
