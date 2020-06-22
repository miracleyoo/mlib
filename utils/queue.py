#!/usr/bin python3
""" Utilities for queues. """

import queue as Queue
import threading


class BackgroundGenerator(threading.Thread):
    """ Run a queue in the background. 
    From:
        https://stackoverflow.com/questions/7323664/python-generator-pre-fetch 
    """

    def __init__(self, generator, prefetch=1):  # See below why prefetch count is flawed
        threading.Thread.__init__(self)
        self.queue = Queue.Queue(prefetch)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self):
        """ Put until queue size is reached.
            Note: put blocks only if put is called while queue has already
            reached max size => this makes 2 prefetched items! One in the
            queue, one waiting for insertion! """
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def iterator(self):
        """ Iterate items out of the queue """
        while True:
            next_item = self.queue.get()
            if next_item is None:
                break
            yield next_item
