"""Test utility functions.
"""
from typing import Iterable


def equivalent(iterable_one: Iterable, iterable_two: Iterable) -> bool:
    if itersize(iterable_one) != itersize(iterable_two):
        return False
    for item in iterable_one:
        if not is_present(item, iterable_two):
            return False
    return True


def itersize(iterable: Iterable) -> int:
    size = 0
    for item in iterable:
        size += 1
    return size


def is_present(item, iterable: Iterable) -> bool:
    for _item in iterable:
        if item == _item:
            return True
    return False  

def qdata(queue) -> list:
    data = list()
    while not queue.empty():
        data.append(queue.get())
    return data
