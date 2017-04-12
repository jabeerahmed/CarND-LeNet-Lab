import time

class Timer:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        if self.verbose: print('elapsed time: %f ms' % self.msecs)
        
    def printTime(self):
        print('elapsed time: {:5.3f} secs | {:10.3f} ms'.format(self.secs, self.msecs))


    def run(func):
        ret = None
        with Timer() as timer: ret = func()
        timer.printTime()
        return ret                   