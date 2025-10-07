from contextlib import contextmanager
import time
class Timer:
    def __init__(self): self.t=time.time()
    def reset(self): self.t=time.time()
    def elapsed(self): return time.time()-self.t

@contextmanager
def time_block(msg):
    t=Timer(); print(f"[TIME] {msg} ...", flush=True)
    yield
    print(f"[TIME] {msg} done in {t.elapsed():.2f}s", flush=True)
