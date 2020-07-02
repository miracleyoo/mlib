import sys


class ExceptionHook:
    """ Send you into a IPython shell when programe crash.
    Usage:
        At any position of your project main file:
        `import mlib.debug.crash_on_ipy`
    From:
        https://www.zhihu.com/question/21572891
    """
    instance = None

    def __call__(self, *args, **kwargs):
        if self.instance is None:
            from IPython.core import ultratb
            self.instance = ultratb.FormattedTB(mode='Plain',
                                                color_scheme='Linux', call_pdb=1)
        return self.instance(*args, **kwargs)


sys.excepthook = ExceptionHook()