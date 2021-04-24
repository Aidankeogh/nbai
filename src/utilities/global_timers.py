import time
import yaml
from collections import defaultdict

class timer:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.calls = 0
        self.total_time = 0

        self.start_time = None
    
    def start(self):
        self.calls += 1
        self.start_time = time.time()
    
    def stop(self):
        self.total_time += time.time() - self.start_time
        self.start_time = None
    
    def sync(self):
        if self.start_time is not None:
            self.total_time += time.time() - self.start_time
            self.start_time = time.time()

class timer_factory:
    def __init__(self):
        self.timers = defaultdict(timer)
        self.last_times = defaultdict(float)
        self.last_calls = defaultdict(int)
    
    def __getitem__(self, key):
        return self.timers[key]
    
    def total(self) -> str:
        print_dict = {}
        for k, v in self.timers.items():
            v.sync()
            print_dict[k] = f"{v.total_time} in {v.calls} calls"
        return yaml.dump(print_dict, default_flow_style=False)
    
    def delta(self) -> str:
        print_dict = {}
        for k, v in self.timers.items():
            v.sync()
            delta = v.total_time - self.last_times[k]
            delta_calls = v.calls - self.last_calls[k]
            if delta > 0:
                print_dict[k] = f"{delta} in {delta_calls} calls"
                self.last_times[k] = float(v.total_time)
                self.last_calls[k] = float(v.calls)
        return yaml.dump(print_dict, default_flow_style=False)
    

timers = timer_factory()

def timeit(func):
    def wrap(*args, **kwargs):
        timers[func.__name__].start()
        result = func(*args, **kwargs)
        timers[func.__name__].stop()
        return result
    return wrap

if __name__ == "__main__":
    timers["hello"].start()
    timers["hello"].stop()
    print(timers.total())
    print(timers.delta())
    timers["goodbye"].start()
    timers["goodbye"].stop()
    print(timers.total())
    print(timers.delta())