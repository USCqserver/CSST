from email.policy import default
import pickle, os, pprint, argparse, sys, warnings, atexit, json
from pathlib import Path

import numpy as np
from tqdm import tqdm

import multiprocessing as mp
from threadpoolctl import threadpool_limits
from numpy.lib.format import open_memmap

from utils.cs_utils import *
from utils.pauli_utils import *
from utils.shadow_utils import *

# ====== Color Preamble ======
RESET   = "\033[0m"
RED     = "\033[91m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
CYAN    = "\033[96m"
# ============================

_FLUSH_EVERY_CS = 1024

_mm = {}
_flush = {}

######### DATA LOGGING FUNCTIONS ##########
# numpy scalar -> python scalar (for json)
to_builtin = lambda x: x.item() if hasattr(x, "item") else x

def reserve_run_id_and_write_params(runs_dir: Path, params: dict, start: int = 1) -> int:
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_id = start
    while True:
        p = runs_dir / f"params_{run_id:04d}.json"
        try:
            with p.open("x") as f:  # atomic create
                json.dump(params, f, indent=2, sort_keys=True)
                f.write("\n")
            return run_id
        except FileExistsError:
            run_id += 1

def write_run_params(runs_dir: Path, run_id: int, params: dict) -> Path:
    p = runs_dir / f"params_{run_id:04d}.json"
    # flatten + clean numpy scalars if any
    clean = {k: to_builtin(v) for k, v in params.items()}
    p.write_text(json.dumps(clean, indent=2, sort_keys=True) + "\n")
    return

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
    
def _init_worker_cs(all_mm_data, times, times_subs, alphas, rescale, axis, inverse, fintint, replace):
    global _times, _times_subs, _alphas, _rescale, _axis, _inverse, _fitint, _replace
    _times, _times_subs, _alphas, _rescale, _axis, _inverse, _fitint, _replace = times, times_subs, alphas, rescale, axis, inverse, fintint, replace
    threadpool_limits(limits=1)
    for mm_data in all_mm_data:
        _init_mm(*mm_data)
    return

def _worker_cs(args):
    ii, jj, kk, ll = args
    m = _times_subs[kk]
    recon, _ = basis_pursuit(times=_times, 
                             num_samps=m, 
                             data=_mm['est'][:,ii,jj], 
                             alpha=_alphas[ll], 
                             seed=42, 
                             rescale=_rescale, 
                             axis=_axis,
                             inverse=_inverse,
                             fit_intercept=_fitint,
                             replace=_replace)
    
    _mm['cs'][ii,jj,kk,ll,:] = get_errors(recon, _mm['exact'][:,ii])

    _flush['cs']['cnt'] += 1
    if _flush['cs']['cnt'] % _flush['cs']['every'] == 0:
        _mm['cs'].flush()
    return

def get_parser():
    parser = argparse.ArgumentParser(description="Command line arguments for CSST")
    parser.add_argument("--dir",    type=str,     default="data",       help="directory to save data files")
    parser.add_argument("--nx",     type=int,     default=2,            help="Number of sites in x direction")
    parser.add_argument("--ny",     type=int,     default=2,            help="Number of sites in y direction")
    parser.add_argument("--ham",    type=str,     default='heis',       help="Hamiltonian type",        choices=['tfim','heis','random'])
    parser.add_argument("--istate", type=str,     default='+-+-',       help="Initial state")
    parser.add_argument("--nw",     type=int,     default=1,            help="Number of workers for multiprocessing")
    #new args
    parser.add_argument("--mmin",   type=int,     default=10,           help="Min m")
    parser.add_argument("--mmax",   type=int,     default=100,          help="Max m")
    parser.add_argument("--mnum",   type=int,     default=3,            help="Number of ms (logspaced)")
    parser.add_argument("--amin",   type=int,     default=-4,           help="Min alpha power (10^amin)")
    parser.add_argument("--amax",   type=int,     default=-1,           help="Max alpha power (10^amax)")
    parser.add_argument("--anum",   type=int,     default=3,            help="Number of alphas (logspaced)")
    parser.add_argument("--snr",    type=float,   default=-1,           help="Filter out (O_i, N_ST) with SNR < value")
    parser.add_argument("--rescale",type=int,     default=0,            help="Rescale by sqrt(N/m) in CS reconstruction", choices=[0,1])
    parser.add_argument("--axis",   type=int,     default=1,            help="Axis to apply DCT/IDCT to", choices=[0,1])
    parser.add_argument("--inverse",type=int,     default=0,            help="Use IDCT instead of DCT for basis", choices=[0,1])
    parser.add_argument("--fitint", type=int,     default=0,            help="Fit intercept in Lasso regression", choices=[0,1])
    parser.add_argument("--replace",type=int,     default=0,            help="Sample with replacement in time-subsampling", choices=[0,1])
    return parser

######### MAIN ##########

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    pprint.pprint(args)

    NX = args.nx
    NY = args.ny
    NQ = NX * NY
    HAM = args.ham
    ISTATE = args.istate
    NUM_WORKERS = args.nw
    MMIN = args.mmin
    MMAX = args.mmax
    MNUM = args.mnum
    ALPHA_MIN = args.amin
    ALPHA_MAX = args.amax
    ALPHA_NUM = args.anum
    SNR_CUTOFF = args.snr
    RESCALE = bool(args.rescale)
    AXIS = args.axis
    INVERSE = bool(args.inverse)
    FITINT = bool(args.fitint)
    REPLACE = bool(args.replace)

    LABEL = f"{NX}x{NY}_{HAM}_{ISTATE}"
    DIR = Path(args.dir) / LABEL
    DIR.mkdir(parents=True, exist_ok=True)

    assert ISTATE in ['ghz','w','r','rp','hr','hrp'] or (len(ISTATE) == NQ and set(ISTATE).issubset(['0','1','+','-','>','<']))
    
    with open(DIR / "info.json", 'r') as handle:
        info = json.load(handle)

    #make sure what user has specified matches existing info json
    assert info['nx'] == NX
    assert info['ny'] == NY
    assert info['ham'] == HAM
    assert info['istate'] == ISTATE

    NB = info['nb']
    ostrings = pbw(NQ, nb=NB, max=True)
    NUM_PAULIS = len(ostrings)
    NSMIN = info['nsmin']
    NSMAX = info['nsmax']
    NSNUM = info['nsnum']
    shadow_subs = np.round(np.logspace(np.log10(NSMIN),np.log10(NSMAX),NSNUM)).astype(int)

    TMIN = info['tmin']
    TMAX = info['tmax']
    N = info['n']
    times = np.linspace(TMIN, TMAX, N)
    assert MMIN >= 1 and MMAX <= N and MNUM >= 1 and MMIN <= MMAX

    times_subs = np.round(np.logspace(np.log10(MMIN),np.log10(MMAX),MNUM)).astype(int)
    alphas = np.logspace(ALPHA_MIN, ALPHA_MAX, ALPHA_NUM, base=10)

    if ((AXIS==0 and INVERSE is False) or (AXIS==1 and INVERSE is True)):
        print(f"{RED}Warning: Mixing synthesis and analysis axes{RESET}")

    params = {"nw": NUM_WORKERS,
              "mmin": MMIN,
              "mmax": MMAX,
              "mnum": MNUM,
              "amin": ALPHA_MIN,
              "amax": ALPHA_MAX,
              "anum": ALPHA_NUM,
              "snr": SNR_CUTOFF,
              "rescale": RESCALE,
              "axis": AXIS,
              "inverse": INVERSE,
              "fitint": FITINT,
              "replace": REPLACE
              }

    ########## MP SETUP OUTSIDE MAIN ##########
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    threadpool_limits(limits=1)
    ###########################################

    exact_path = DIR / "exacts.npy"
    assert exact_path.exists(), f"Exacts file {exact_path} does not exist - create it first with DATA_GEN.py"
    exacts = np.load(exact_path, mmap_mode='r')
    print(f"{'Size of exacts:':<20} {exacts.nbytes / (1024**2):8.2f} MB")    

    err_path = DIR / "errs.npy"
    assert err_path.exists(), f"Errors file {err_path} does not exist - create it first with DATA_GEN.py"
    errs = np.load(err_path, mmap_mode='r')
    print(f"{'Size of errs:':<20} {errs.nbytes / (1024**2):8.2f} MB")

    est_path = DIR / "ests.npy"
    assert est_path.exists(), f"Estimates file {est_path} does not exist - create it first with DATA_GEN.py"
    ests = np.load(est_path, mmap_mode='r')
    print(f"{'Size of ests:':<20} {ests.nbytes / (1024**2):8.2f} MB")

    RUN_DIR = DIR / "runs"
    run_id = reserve_run_id_and_write_params(RUN_DIR, params)
    cs_path = RUN_DIR / f"cs_{run_id:04d}.npy"
    cs_errs = create_memmap(cs_path, shape=(NUM_PAULIS,NSNUM,MNUM,ALPHA_NUM,8), dtype=np.float32, fill=np.nan)
    print(f"{'Size of cs_errs:':<20} {cs_errs.nbytes / (1024**2):8.2f} MB")
    print(f"Run id: {run_id:04d}\n")

    snrs = errs[:,:,6].flatten()
    if SNR_CUTOFF is not None:
        percent = (np.sum(snrs > SNR_CUTOFF) / snrs.size) * 100
        print(f"{percent:.2f}% of estimates have SNR > {SNR_CUTOFF} dB")
        print(f"{GREEN}Filtering low SNR observables{RESET}")
    else:
        print(f"{YELLOW}Not filtering low SNR observables{RESET}")

    def slurm_cpus():
        v = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_CPUS_ON_NODE")
        return max(1, int(v)) if v else (os.cpu_count() or 1)
    
    NUM_WORKERS = 6 if (sys.platform == "darwin") else min(slurm_cpus(), NUM_WORKERS)
    print(f"{CYAN}Using multiprocessing with {NUM_WORKERS} workers...{RESET}")

    with mp.Pool(processes=NUM_WORKERS,
                 initializer=_init_worker_cs,
                 initargs=([('exact', exact_path, 'r', None),
                            ('est', est_path, 'r', None),
                            ('cs', cs_path, 'r+', _FLUSH_EVERY_CS)],
                            times,
                            times_subs,
                            alphas,
                            RESCALE,
                            AXIS,
                            INVERSE,
                            FITINT,
                            REPLACE)) as pool:

        #NOTE: using np.nan to flag entries with low SNR, thus I can't also use np.nan to flag uncomputed entries...
        task_iter = (
            (ii, jj, kk, ll)
            for ii in range(NUM_PAULIS)
            for jj in range(NSNUM)
            if (SNR_CUTOFF is None) or (errs[ii, jj, 6] > SNR_CUTOFF)
            for kk in range(MNUM)
            for ll in range(len(alphas))
            )

        total = sum(
            1
            for ii in range(NUM_PAULIS)
            for jj in range(NSNUM)
            if (SNR_CUTOFF is None) or (errs[ii, jj, 6] > SNR_CUTOFF)
            ) * MNUM * len(alphas)

        for _ in tqdm(pool.imap_unordered(_worker_cs, task_iter, chunksize=128),
                                    total=total,
                                    desc=f"{'running compressed sensing':<35}",
                                    miniters=max(1, total // 100),
                                    mininterval=1.0,
                                    file=sys.stdout, leave=True, disable=False): #seconds
            pass #workers write directly to memmap

    cs_errs.flush()

