import time

def time_block(fn, *args, **kwargs):
    t0 = time.time()
    out = fn(*args, **kwargs)
    dt = time.time() - t0
    return out, dt
