import tkinter as tk
import pickle as pkl
import os
import sys
import threading
import inspect
import ctypes
import subprocess
from tkinter import filedialog, messagebox
from functools import partial
from PIL import Image, ImageTk
from pathlib2 import Path

__all__ = ["Window", "tk", "argStation", "dotdict"]
apath = Path(os.path.dirname(os.path.realpath(__file__)))


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def _async_raise(tid, exctype):
    '''Raises an exception in the threads with id tid'''
    if not inspect.isclass(exctype):
        raise TypeError("Only types can be raised (not instances)")
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid),
                                                     ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # "if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), None)
        raise SystemError("PyThreadState_SetAsyncExc failed")

class ThreadWithExc(threading.Thread):
    '''A thread class that supports raising exception in the thread from
       another thread.
    '''
    def _get_my_tid(self):
        """determines this (self's) thread id

        CAREFUL : this function is executed in the context of the caller
        thread, to get the identity of the thread represented by this
        instance.
        """
        if not self.isAlive():
            raise threading.ThreadError("the thread is not active")

        # do we have it cached?
        if hasattr(self, "_thread_id"):
            return self._thread_id

        # no, look for it in the _active dict
        for tid, tobj in threading._active.items():
            if tobj is self:
                self._thread_id = tid
                return tid

        # TODO: in python 2.6, there's a simpler way to do : self.ident

        raise AssertionError("could not determine the thread's id")

    def raiseExc(self, exctype):
        """Raises the given exception type in the context of this thread.

        If the thread is busy in a system call (time.sleep(),
        socket.accept(), ...), the exception is simply ignored.

        If you are sure that your exception should terminate the thread,
        one way to ensure that it works is:

            t = ThreadWithExc( ... )
            ...
            t.raiseExc( SomeException )
            while t.isAlive():
                time.sleep( 0.1 )
                t.raiseExc( SomeException )

        If the exception is to be caught by the thread, you need a way to
        check that your thread has caught it.

        CAREFUL : this function is executed in the context of the
        caller thread, to raise an excpetion in the context of the
        thread represented by this instance.
        """
        _async_raise( self._get_my_tid(), exctype )

class CreateToolTip(object):
    """
    create a tooltip for a given widget
    """

    def __init__(self, widget, text='widget info'):
        self.waittime = 500  # miliseconds
        self.wraplength = 180  # pixels
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)
        self.widget.bind("<ButtonPress>", self.leave)
        self.id = None
        self.tw = None

    def enter(self, event=None):
        self.schedule()

    def leave(self, event=None):
        self.unschedule()
        self.hidetip()

    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(self.waittime, self.showtip)

    def unschedule(self):
        id = self.id
        self.id = None
        if id:
            self.widget.after_cancel(id)

    def showtip(self, event=None):
        x = y = 0
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        # creates a toplevel window
        self.tw = tk.Toplevel(self.widget)
        # Leaves only the label and removes the app window
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(self.tw, text=self.text, justify='left',
                         background="#ffffff", relief='solid', borderwidth=1,
                         wraplength=self.wraplength)
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tw
        self.tw = None
        if tw:
            tw.destroy()
            

class Window(tk.Frame):
    """
    Here, we are creating our class, Window, and inheriting from the Frame
    class. Frame is a class from the tkinter module. (see Lib/tkinter/__init__)
    """
    # Define settings upon initialization. Here you can specify
    def __init__(self, master=None, parser=None, main_func="main.py", one_off=True):
        
        # parameters that you want to send through the Frame class. 
        tk.Frame.__init__(self, master)   

        #reference to the master widget, which is the tk window                 
        self.master = master
        self.main_func = main_func
        self.one_off =one_off

        if parser is None:
            raise KeyError("You need to pass in a parser to start the window!")
        else:
            parser.sort()
            self.parser = parser

        #with that, we want to then run init_window, which doesn't yet exist
        self.init_window()

    #Creation of init_window
    def init_window(self):
        # changing the title of our master widget
        self.master.title(self.parser.title)

        # allowing the widget to take the full space of the root window
        self.pack(fill=tk.BOTH, expand=1)

        #####################
        # -- CREATE MENU -- #
        #####################

        # creating a menu instance
        menu = tk.Menu(self.master)
        self.master.config(menu=menu)

        # create the file object
        file = tk.Menu(menu)

        # adds a command to the menu option, calling it exit, and the
        # command it runs on event is client_exit
        file.add_command(label="Exit", command=self.client_exit)
        file.add_command(label="Info", command=self.show_info)

        #added "file" to our menu
        menu.add_cascade(label="Help", menu=file)

        #################
        # -- MY PART -- #
        #################
        # text = Label(self, text="Hey there good lookin!")
        # btn_input_path = Button(self, text="Choose the input file path.")
        self.text_items = {}
        self.input_items = {}
        self.value_items = {}
        for i, arg_name in enumerate(list(self.parser.arguments.keys())):
            self.text_items[arg_name] = self.make_element(
                tk.Label, 2, 2+30*i, 25, 150, text=arg_name+":")  # Label(self, text=arg_name+":")

            # -- Add Question Mark Hover Help -- #
            load = Image.open(str(apath / 'img' / 'question_mark.png'))
            render = ImageTk.PhotoImage(load.resize((20, 20), Image.ANTIALIAS))
            img = self.make_element(
                tk.Label, 670, (2+30*i), 20, 20, image=render)
            img.image = render
            CreateToolTip(img, text=self.parser.arguments[arg_name].help)

            # -- Create Label, Input Entry (and Open Button) -- #
            if self.parser.arguments[arg_name].type == str:
                if self.parser.arguments[arg_name].is_path or self.parser.arguments[arg_name].is_dir:
                    self.value_items[arg_name] = tk.StringVar()
                    self.value_items[arg_name].set(
                        "" if self.parser.arguments[arg_name].default == None else self.parser.arguments[arg_name].default)
                    self.input_items[arg_name] = self.make_element(
                        tk.Entry, 160, 2+30*i, 25, 450, textvariable=self.value_items[arg_name], width=200)
                    if self.parser.arguments[arg_name].is_path:
                        button = self.make_element(tk.Button, 615, 2+30*i, 25, 45, text="Open", command=partial(
                            self.show_file_selector, self.value_items, arg_name))
                    else:
                        button = self.make_element(tk.Button, 615, 2+30*i, 25, 45, text="Open", command=partial(
                            self.show_dir_selector, self.value_items, arg_name))
                else:
                    self.value_items[arg_name] = tk.StringVar()
                    self.value_items[arg_name].set(
                        "" if self.parser.arguments[arg_name].default == None else self.parser.arguments[arg_name].default)
                    self.input_items[arg_name] = self.make_element(
                        tk.Entry, 160, 2+30*i, 25, 500, textvariable=self.value_items[arg_name], width=200)
            elif self.parser.arguments[arg_name].type == bool:
                self.value_items[arg_name] = tk.BooleanVar()
                self.make_element(tk.Radiobutton, 160, 2+30*i, 25, 100,
                                  text='True', variable=self.value_items[arg_name], value=True)
                self.make_element(tk.Radiobutton, 270, 2+30*i, 25, 100,
                                  text='False', variable=self.value_items[arg_name], value=False)
            elif self.parser.arguments[arg_name].type == float:
                self.value_items[arg_name] = tk.DoubleVar()
                self.value_items[arg_name].set(0.0 if self.parser.arguments[arg_name].default == None else float(
                    self.parser.arguments[arg_name].default))
                self.input_items[arg_name] = self.make_element(
                    tk.Entry, 160, 2+30*i, 25, 500, textvariable=self.value_items[arg_name], width=200)
            elif self.parser.arguments[arg_name].type == int:
                self.value_items[arg_name] = tk.IntVar()
                self.value_items[arg_name].set(0 if self.parser.arguments[arg_name].default == None else int(
                    self.parser.arguments[arg_name].default))
                self.input_items[arg_name] = self.make_element(
                    tk.Entry, 160, 2+30*i, 25, 500, textvariable=self.value_items[arg_name], width=200)
            else:
                raise ValueError(arg_name+" has an unsupported type!")

        # -- Create Start and End Button -- #
        self.main_proc=None
        # self.main_thread.daemon = True
        self._btn_start = self.make_element(
            tk.Button, 15, 22+30*(i+1), 35, 250, expand=1, text="Start", command=self.start_program)
        self._btn_stop = self.make_element(
            tk.Button, 435, 22+30*(i+1), 35, 250, expand=1, text="Stop", command=self.stop_program)
        self.geometry = "700x{}".format(30*(i+2)+33)

    # When you click the start button, we start the main program
    def start_program(self):
        if not os.path.exists("./.gui/"):
            os.makedirs("./.gui/", exist_ok=True)
        with open("./.gui/args.pkl","wb") as f:
            temp_args = {k: v.get() for k, v in self.value_items.items()}
            pkl.dump(temp_args,f)
        self.main_proc = subprocess.Popen("python "+self.main_func)#, shell = True)
        print("Main process ID: ", self.main_proc)
        # self.main_thread = ThreadWithExc(target=lambda:(
        # self.main_func(dotdict({k: v.get()
        #     for k, v in self.value_items.items()})),
        # messagebox.showinfo("Info", "Main function execution finished!")))
        # self.main_thread.start()

    def stop_program(self):
        if self.main_proc is None:
            sys.exit(0)
        else:
            self.main_proc.kill()
            self.main_proc=None
            if self.one_off:
                sys.exit(0)                

    # Make a tkinter component with a (x,y) position and height, width
    def make_element(self, element_type, x, y, h, w, expand=None, *args, **kwargs):
        f = tk.Frame(self, height=h, width=w)
        f.pack_propagate(0)  # don't shrink
        f.place(x=x, y=y)
        label = element_type(f, *args, **kwargs)
        label.pack(fill=tk.BOTH, side='left', expand=expand)
        return label

    # Show a file selector and pass the selected path to value_item
    def show_file_selector(self, value_items, arg_name):
        file_path = filedialog.askopenfilename(initialdir=os.getcwd(),
            title="Select file", filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
        if file_path is not None and file_path != "":
            value_items[arg_name].set(file_path)
    
    def show_dir_selector(self, value_items, arg_name):
        dir_path = filedialog.askdirectory(initialdir=os.getcwd(), title="Select directory")
        if dir_path is not None and dir_path != "":
            value_items[arg_name].set(dir_path)

    # Exit the client
    def client_exit(self):
        sys.exit(0)

    # Show the help information
    def show_info(self):
        messagebox.showinfo(
            "Info", "This GUI program is built by Zhongyang Zhang(Miracleyoo), in Jan 25, 2020."+
            "Hope you have fun using it. If you have any question, please contact mirakuruyoo@gmail.com.")


class argInfoPack():
    """
    The information pack for a certain entry.
    """
    def __init__(self, name, type, default=None, action=None, is_path=False, is_dir=False, help=None):
        super().__init__()
        self.name = name
        self.type = type
        self.default = default
        self.is_path = is_path
        self.is_dir = is_dir
        self.help = help

class argStation():
    """
    The parser. It only support limited parts of argparse, and decode them into argInfoPack.
    """
    def __init__(self, title="Miracleyoo GUI"):
        super().__init__()
        self.arguments = {}
        self.title = title
    
    # Add one argument to arguments.
    def add_argument(self, *name, type=str, default=None, action=None, is_path=False, is_dir=False, help="No help message for this item."):
        if len(name)==0:
            print("Please at least input a name for this argument!")
        elif len(name)==1:
            name = name[0].lstrip("-").strip()
        else:
            for n in name:
                if n.startswith("--"):
                    name=n.lstrip("-").strip()
                    break
                elif n.startswith("-"):
                    name=n.lstrip("-").strip()
                else:
                    name=n.strip()
                    break
        if action=="store_true": type=bool
        self.arguments[name]=argInfoPack(name=name,type=type,default=default, action=action, is_path=is_path, is_dir=is_dir, help=help)
    
    # Sort the argument dictionary by its type and action.
    def sort(self):
        def compute_order(arg):
            if arg.type == str:
                if arg.is_path or arg.is_dir:
                    return 0
                else:
                    return 1
            elif arg.type == bool:
                return 3
            else:
                return 2
        arguments = {k: v for k, v in sorted(self.arguments.items(), key=lambda item: compute_order(item[1]))}
        self.arguments = arguments


"""
# The example code you should write in your own code.
parser = argStation("Miracleyoo Test Program")
parser.add_argument(
    '--root_path',
    type=str,
    is_path=True,
    default="./Data/Mixer_Videos/",
    help='The root path of recorded videos.')
root = tk.Tk()
#creation of an instance
app = Window(root, parser, main_func="main.py")
root.geometry(app.geometry)
#mainloop 
root.mainloop()  
"""
