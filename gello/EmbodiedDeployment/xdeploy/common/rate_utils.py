import time


class Rate:
    def __init__(self, hz):
        self.period = 1.0 / hz
        self.last_time = time.time()

    def sleep(self):
        now = time.time()
        elapsed = now - self.last_time
        sleep_time = self.period - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        self.last_time = time.time()
