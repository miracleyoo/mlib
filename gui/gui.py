from .guiparse import *
import pickle as pkl

__all__ = ['start_gui', 'load_args']


def start_gui(parser, main_func, one_off=False):
    root = tk.Tk()
    # creation of an instance
    app = Window(
        root, parser, main_func=main_func, one_off=False)
    root.geometry(app.geometry)
    # mainloop
    root.mainloop()


def load_args():
    with open("./.gui/args.pkl", "rb") as f:
        args = dotdict(pkl.load(f))
    return args
