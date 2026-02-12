from pathlib import Path
import pickle, os, pprint, argparse, sys, warnings, atexit, json

import networkx
import numpy as np
import qutip as qt
from tqdm import tqdm

import multiprocessing as mp
from multiprocessing import shared_memory
from threadpoolctl import threadpool_limits
from numpy.lib.format import open_memmap

from utils.cs_utils import *
from utils.pauli_utils import *
from utils.shadow_utils import *

# ====== color preamble ======
RESET   = "\033[0m"
RED     = "\033[91m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
CYAN    = "\033[96m"
# ============================

_FLUSH_EVERY_SHADOWS = 16
_FLUSH_EVERY_EST = 64

_mm = {}
_flush = {}

######### MP FUNCTIONS ##########
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

def _init_mm(name: str, path: str, mmap_mode: str, flush_every: int | None = None):
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

def _init_states(shm_name, shape, dtype_str):
    global _shm_states, _states
    _shm_states = shared_memory.SharedMemory(name=shm_name, create=False)  # keep reference!
    _states = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=_shm_states.buf)
    _states.setflags(write=False)
    atexit.register(_shm_states.close)  # parent should unlink() when all done
    return

def _init_ops(ostrings):
    global _ops, _NQ, _ostrings
    _NQ = len(ostrings[0])
    _ops = [p2op(x, norm=False) for x in ostrings]
    _ostrings = ostrings
    return

######################
def _init_worker_exact(state_data, ostrings):
    threadpool_limits(limits=1)
    _init_states(*state_data)
    _init_ops(ostrings)
    return

def _worker_exact(itt):
    rho = qt.Qobj(_states[itt], dims=[[2]*_NQ]*2, copy=False)
    rho = rho if rho.isherm else (rho + rho.dag())/2  # enforce hermiticity
    expvals = qt.expect(_ops, rho)
    expvals = np.real_if_close(expvals, tol=1e8) # drops imag if |Im| <= 10^-12-ish
    # print(np.max(np.abs(np.imag(expvals))))
    return itt, expvals

######################
def _init_worker_shadow(state_data, all_mm_data, nq, nsmax):
    threadpool_limits(limits=1)
    global _NQ, _NSMAX
    _NQ, _NSMAX = nq, nsmax
    _init_states(*state_data)
    for mm_data in all_mm_data:
        _init_mm(*mm_data)
    return

def _worker_shadow(args):
    itt, seed = args
    rho = qt.Qobj(_states[itt], dims=[[2]*_NQ]*2, copy=False)
    rho = rho if rho.isherm else (rho + rho.dag())/2  # enforce hermiticity

    _mm['shadow'][itt] = get_shadows(rho, _NSMAX, seed=seed)  # (NSMAX, Nq, 2)
    _flush['shadow']['cnt'] += 1
    if _flush['shadow']['cnt'] % _flush['shadow']['every'] == 0:
        _mm['shadow'].flush()
    return

######################
def _init_worker_est(all_mm_data, ostrings, shadow_subs):
    global _shadow_subs
    _shadow_subs = shadow_subs
    threadpool_limits(limits=1)
    for mm_data in all_mm_data:
        _init_mm(*mm_data)
    _init_ops(ostrings)
    return

def _worker_est(itt):
    _mm['est'][itt] = estimate_batch(_mm['shadow'][itt], _ostrings, _shadow_subs)
    _flush['est']['cnt'] += 1
    if _flush['est']['cnt'] % _flush['est']['every'] == 0:
        _mm['est'].flush()
    return

######################
def get_parser():
    parser = argparse.ArgumentParser(description="Command line arguments for CSST")
    parser.add_argument("--dir",    type=str,     default="data",       help="directory to save data files")
    parser.add_argument("--nx",     type=int,     default=2,            help="Number of sites in x direction")
    parser.add_argument("--ny",     type=int,     default=2,            help="Number of sites in y direction")
    parser.add_argument("--nb",     type=int,     default=4,            help="Max Pauli observable weight")
    parser.add_argument("--ham",    type=str,     default='heis',       help="Hamiltonian type",        choices=['tfim','heis'])
    parser.add_argument("--istate", type=str,     default='+-+-',       help="Initial state ('ghz','w','r','rp','hr','hrp', or length NQ string of [0,1,+,-,>,<])")
    parser.add_argument("--n",      type=int,     default=500,          help="Number of time steps")
    parser.add_argument("--nsmin",  type=int,     default=10,           help="Min number of shadows")
    parser.add_argument("--nsmax",  type=int,     default=1000,         help="Max number of shadows")
    parser.add_argument("--nsnum",  type=int,     default=10,           help="Number of shadows to try (logspaced)")
    parser.add_argument("--nw",     type=int,     default=1,            help="Number of workers for multiprocessing")
    return parser

if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()
    pprint.pprint(args)

    NX = args.nx               # lattice dimension x
    NY = args.ny               # lattice dimension y
    NQ = NX * NY               # number of qubits
    N = args.n                 # number of time steps
    NB = args.nb               # max Pauli observable weight
    HAM = args.ham             # Hamiltonian type: 'tfim', 'random', 'heis'
    NSMIN = args.nsmin         # min number of shadows
    NSMAX = args.nsmax         # max number of shadows
    NSNUM = args.nsnum         # number of shadow subsamples
    ISTATE = args.istate       # initial state
    NUM_WORKERS = args.nw      # number of workers

    LABEL = f"{NX}x{NY}_{HAM}_{ISTATE}"
    DIR = Path(args.dir) / LABEL
    print(f"Creating directory {DIR}")
    DIR.mkdir(parents=True, exist_ok=True)

    assert ISTATE in ['ghz','w','r','rp','hr','hrp'] or (len(ISTATE) == NQ and set(ISTATE).issubset(['0','1','+','-','>','<']))
    shadow_subs = np.round(np.logspace(np.log10(NSMIN),np.log10(NSMAX),NSNUM)).astype(int)

    ########## MP SETUP OUTSIDE MAIN ##########
    # spawn is portable (macOS default). Works fine on Ubuntu too.
    if sys.platform == "darwin":
        mp.set_start_method("spawn", force=True)
        
    # try:
        # mp.set_start_method("spawn", force=True)
    # except RuntimeError:
        # pass
    threadpool_limits(limits=1)

    ########## MODEL ##########
    G = networkx.generators.lattice.grid_2d_graph(NX,NY)
    G = networkx.convert_node_labels_to_integers(G)
    H_ops = get_h_ops(NQ, model=HAM, graph=G, seed=42)

    c_ops = []
    ad_gammas = [1e-1]*NQ
    ad_ops = amp_damp_ops(NQ)
    c_ops += [gamma * op for (gamma,op) in zip(ad_gammas, ad_ops)]

    dz_gammas = [1e-1]*NQ
    dz_ops = [p2op(z) for z in pbw(NQ,nb=1,ptype='Z')]
    c_ops += [gamma * op for (gamma,op) in zip(dz_gammas, dz_ops)]

    re_bound, im_bound = get_eigval_bounds_loose(H_ops, ad_gammas + dz_gammas, ub=1)
    # print(f"\nReal part bound: {re_bound}, Imaginary part bound: {im_bound}")
    dt_nyq = np.pi/im_bound
    dt = dt_nyq/4

    H = ps2op(H_ops)
    L = qt.liouvillian(H,c_ops=c_ops)

    ########## SIMULATION PARAMETERS ##########
    MAXT = dt * N
    times = np.linspace(0,MAXT,N)

    init_state = get_init_state(NQ, ISTATE)

    ostrings = pbw(NQ, nb=NB, max=True)
    # ostrings.remove("I"*NQ)
    NUM_PAULIS = len(ostrings)

    # ODE solver options
    options = {'nsteps':1000,           #<-- make this bigger if you get an ODE error
               'progress_bar':True,
               'atol':1e-9,
               'rtol':1e-7
               }

    ########## MESOLVE SIMULATION ##########
    print("\nStarting mesolve...")
    result = qt.mesolve(H, 
                        rho0=init_state, 
                        tlist=times, 
                        c_ops=c_ops, 
                        e_ops=[], 
                        options=options) #<-- if e_ops == [], then the density matrix is returned at each time
    print("\n")
    STATES = list(result.states) 
    print(f"{'Size of states:':<20} {STATES.__sizeof__() / (1024**2):8.2f} MB")    
    ########################################
    
    #make rng
    rng = np.random.default_rng(42)

    #data arrays
    exact_path = DIR / "exacts.npy"
    exacts = create_memmap(exact_path, shape=(N, NUM_PAULIS), dtype=np.float32, fill=np.nan)
    print(f"{'Size of exacts:':<20} {exacts.nbytes / (1024**2):8.2f} MB")    

    err_path = DIR / "errs.npy"
    errs = create_memmap(err_path, shape=(NUM_PAULIS, NSNUM, 8), dtype=np.float32, fill=np.nan)  
    print(f"{'Size of errs:':<20} {errs.nbytes / (1024**2):8.2f} MB")    

    shadow_path = DIR / "shadows.npy"
    shadows = create_memmap(shadow_path, shape=(N, NSMAX, NQ, 2), dtype=np.int8, fill=0) #cant put nan in int8
    print(f"{'Size of shadows:':<20} {shadows.nbytes / (1024**2):8.2f} MB")    

    est_path = DIR / "ests.npy"
    ests = create_memmap(est_path, shape=(N, NUM_PAULIS, NSNUM), dtype=np.float32, fill=np.nan)
    print(f"{'Size of ests:':<20} {ests.nbytes / (1024**2):8.2f} MB")

    def cpu_cap():
        v = os.environ.get("SLURM_CPUS_PER_TASK")
        if v:
            return int(v)
        return os.cpu_count() or 1

    NUM_WORKERS = min(cpu_cap(), NUM_WORKERS)

    print(f"{CYAN}Using multiprocessing with {NUM_WORKERS} workers...{RESET}")
    print("\n")

    ########################################
    info = {'nx':     NX,
            'ny':     NY,
            'nb':     NB,
            'ham':    HAM,
            'istate': ISTATE,
            'nsmin':  NSMIN,
            'nsmax':  NSMAX,
            'nsnum':  NSNUM,
            'tmin':   times[0],
            'tmax':   times[-1],
            'n':      N,
            'dt':     dt,
            'nw':     NUM_WORKERS,}
    
    to_builtin = lambda x: x.item() if hasattr(x, "item") else x
    info = {k: to_builtin(v) for k, v in info.items()}
    with (DIR / "info.json").open("w") as f:
        json.dump(info, f, indent=2)
    pprint.pprint(info)    
    print("\n")
    ########################################

    try:
        states_arr = np.stack([np.asarray(temp.full()) for temp in STATES])   # convert to (N, d, d) complex128
        shm = shared_memory.SharedMemory(create=True, size=states_arr.nbytes)   #create shared memory block
        state_data = [shm.name, states_arr.shape, str(states_arr.dtype)]

        shm_view = np.ndarray(states_arr.shape, dtype=states_arr.dtype, buffer=shm.buf)
        shm_view[:] = states_arr
        shm_view.setflags(write=False)
        
        #exacts
        with mp.Pool(processes=NUM_WORKERS,
                     initializer=_init_worker_exact,
                     initargs=(state_data, ostrings,)) as pool:
            
            for itt, row in tqdm(pool.imap_unordered(_worker_exact, range(N), chunksize=16),
                                 total=N, 
                                 desc=f"{'exacts over time':<35}",
                                 mininterval=1,
                                 file=sys.stdout, leave=True, disable=False):
                exacts[itt] = row
        exacts.flush()

        #shadows
        with mp.Pool(processes=NUM_WORKERS,
                     initializer=_init_worker_shadow,
                     maxtasksperchild=50,
                     initargs=(state_data,
                               [['shadow', shadow_path, 'r+', _FLUSH_EVERY_SHADOWS]], 
                               NQ,
                               NSMAX)) as pool:

            task_iter = (
                (itt, int(rng.integers(0, 2**32 - 1, dtype=np.uint32)))
                for itt in range(N)
            )

            for _ in tqdm(pool.imap_unordered(_worker_shadow, task_iter, chunksize=1),
                          total=N,
                          desc=f"{'generating shadows':<35}",
                          miniters=max(1, N // 100),
                          mininterval=1.0,
                          file=sys.stdout, leave=True, disable=False):
                pass #workers write directly to memmap
        shadows.flush()

        #estimates
        task_iter = (
            itt
            for itt in range(N)
            )

        with mp.Pool(processes=NUM_WORKERS,
                     initializer=_init_worker_est,
                     maxtasksperchild=50, #trying to fix hung workers
                     initargs=([['shadow', shadow_path, 'r', None],
                                ['est', est_path, 'r+', _FLUSH_EVERY_EST]], 
                                ostrings,
                                shadow_subs)) as pool:

            for _ in tqdm(pool.imap_unordered(_worker_est, task_iter, chunksize=10),
                          total=N,
                          desc=f"{'estimating shadow batches':<35}",
                          mininterval=1.0,
                          file=sys.stdout, leave=True, disable=False):
                pass #workers write directly to memmap
        ests.flush()

        #errors
        #compute errors in main process since this is fast enough (by now, ests is populated)
        for ii in tqdm(range(NUM_PAULIS), desc=f"{'computing errors':<35}"):
            true = exacts[:,ii].copy()
            for jj in range(NSNUM):
                recon = ests[:,ii,jj].copy()
                errs[ii,jj,:] = get_errors(recon, true)

    finally:
        shm.close()
        shm.unlink()


