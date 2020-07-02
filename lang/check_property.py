""" Provide class with input type check properties.

Usage:
    class Person:
        name = String('name')
        age = Integer('age')

        def __init__(self, name, age):
            self.name = name
            self.age = age

Link:
    https://python3-cookbook.readthedocs.io/zh_CN/latest/c09/p21_avoid_repetitive_property_methods.html
"""

from functools import partial


def typed_property(name, expected_type):
    storage_name = '_' + name

    @property
    def prop(self):
        return getattr(self, storage_name)

    @prop.setter
    def prop(self, value):
        if not isinstance(value, expected_type):
            raise TypeError('{} must be a {}'.format(name, expected_type))
        setattr(self, storage_name, value)

    return prop


String = partial(typed_property, expected_type=str)
Integer = partial(typed_property, expected_type=int)
Float = partial(typed_property, expected_type=float)

