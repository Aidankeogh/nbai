from collections import defaultdict
import pandas as pd
import numpy as np
global_ema_dict = {}

def ema(name, x, a=0.01):
    current = float(x)
    if name not in global_ema_dict:
        global_ema_dict[name] = current
    global_ema_dict[name] = global_ema_dict[name] * (1-a) + current * a
    return  global_ema_dict[name]


if __name__ == "__main__":
    for i in range(30):
        print(ema("test", i))