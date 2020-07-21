""" Access dictionary items using dot operator.
Usage:
    mydict = {'val':'it works'}
    nested_dict = {'val':'nested works too'}
    mydict = dotdict(mydict)
    mydict.val
    # 'it works'

    mydict.nested = dotdict(nested_dict)
    mydict.nested.val
    # 'nested works too'
Urls:
    https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary/28463329
"""

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
