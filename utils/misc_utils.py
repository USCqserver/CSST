import os
import numpy as np
from numpy.lib.format import open_memmap

def create_memmap(mm_path, shape, dtype, fill, order=False):
    """ 
    w+ mode: create or overwrite existing file for reading and writing. If mode == 'w+' then shape must also be specified
    """
    if os.path.exists(mm_path):
        print(Warning(f"File {mm_path} already exists and will be overwritten."))

    mm = open_memmap(mm_path, mode="w+", dtype=dtype, shape=shape, fortran_order=order)
    mm[...] = fill
    mm.flush()
    return mm

_mm = {}
_flush = {}

def init_mm(name: str, path: str, mmap_mode: str, flush_every: int | None = None):
    """
    Register a memmap/array under a name in a global dict.
    mmap_mode: 'r' or 'r+' (or None if you want normal np.load without mmap)
    """
    global _mm, _flush

    arr = np.load(path, mmap_mode=mmap_mode)
    _mm[name] = arr

    if mmap_mode in ("r+", "w+", "w") and flush_every:
        _flush[name] = {"cnt": 0, "every": int(flush_every)}
        # atexit.register(arr.flush)
    return

def cpu_cap():
    """Number of worker processes to use: SLURM's per-task allocation if set, else cpu_count-1."""
    v = os.environ.get("SLURM_CPUS_PER_TASK")
    if v:
        return int(v)
    n = os.cpu_count() or 1
    return max(1, n-1)