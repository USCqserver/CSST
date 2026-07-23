#!/usr/bin/env python
# coding: utf-8

from tqdm import tqdm
import random, os, argparse, pprint, json, sys
import multiprocessing as mp
from pathlib import Path
from threadpoolctl import threadpool_limits

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

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

def get_L(Nq, Nx, Ny, ham, gamma=1e-2, eps=1e-1, seed=42):
    G = nx.generators.lattice.grid_2d_graph(Nx,Ny)
    G = nx.convert_node_labels_to_integers(G)
    H_ops = get_h_ops(Nq, model=ham, graph=G, seed=seed, eps=eps)

    c_ops = []
    rng = np.random.default_rng(seed+1)
    ad_gammas = ([np.sqrt(gamma)]*Nq) * (1 + eps * rng.uniform(low=-1, high=1, size=Nq))  # amplitude damping rates
    ad_ops = amp_damp_ops(Nq)
    c_ops += [gamma * op for (gamma,op) in zip(ad_gammas, ad_ops)]

    dz_gammas = ([np.sqrt(gamma)]*Nq) * (1 + eps * rng.uniform(low=-1, high=1, size=Nq))  # dephasing rates
    dz_ops = [p2op(z) for z in pbw(Nq,nb=1,ptype='Z')]
    c_ops += [gamma * op for (gamma,op) in zip(dz_gammas, dz_ops)]

    H = ps2op(H_ops)
    L = qt.liouvillian(H, c_ops)
    return L

def gini_sparsity(v):
    v = np.abs(v)
    if np.sum(v) == 0:
        return 1.0
    v = np.sort(v)          # ascending
    n = len(v)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * v)) / (n * np.sum(v)) - (n + 1) / n
    return gini / (1 - 1/n)

def hoyer_sparsity(v):
    n = len(v)
    l1 = np.sum(np.abs(v))
    l2 = np.linalg.norm(v)
    if l2 == 0: return 1.0
    return (np.sqrt(n) - l1/l2) / (np.sqrt(n) - 1)

def effective_sparsity(v, tol=1e-1):
    k = np.sum(np.abs(v) > tol)   # number of "active" components
    return 1 - k / len(v)

def get_sparsity(x, type):
    if type == 'gini':
        return gini_sparsity(x)
    elif type == 'hoyer':
        return hoyer_sparsity(x)
    elif type == 'effective':
        return effective_sparsity(x)
    else:
        raise ValueError(f"Unknown sparsity type: {type}")

def cpu_cap():
    v = os.environ.get("SLURM_CPUS_PER_TASK")
    if v:
        return int(v)
    n = os.cpu_count() or 1
    return max(1, n - 1)

def _init_worker(nq, nx, ny, nb, ham, gamma, istate):
    threadpool_limits(limits=1)
    global _NQ, _NX, _NY, _NB, _HAM, _GAMMA, _ISTATE
    _NQ, _NX, _NY, _NB, _HAM, _GAMMA, _ISTATE = nq, nx, ny, nb, ham, gamma, istate

def _worker(args):
    ii, jj, eps, seed = args
    U = COB(_NQ)
    ostrings = pbw(_NQ, nb=_NB, max=True)
    NUM_PAULIS = len(ostrings)
    G = nx.generators.lattice.grid_2d_graph(_NX,_NY)
    G = nx.convert_node_labels_to_integers(G)
    init_state = get_init_state(_NQ, _ISTATE, graph=G)

    L = get_L(_NQ, _NX, _NY, _HAM, eps=eps, gamma=_GAMMA, seed=seed)
    Lp = U.dag() * L * U
    Es, V = np.linalg.eig(Lp.full())
    cn = np.linalg.cond(V)
    if cn > 1e6:
        print(f"Warning: Condition number of V is large ({cn:.2e}) for trial {ii}, eps {eps:.2f}. Results may be inaccurate.")
    Vinv = np.linalg.inv(V)
    state_sparsity = get_sparsity(expand_into(Vinv=Vinv, state=init_state, U=U), type='gini')
    obs_sparsity = np.zeros(NUM_PAULIS, dtype=np.float64)
    for kk in range(NUM_PAULIS):
        obs_sparsity[kk] = get_sparsity(expand_into(V=V, observable=p2op(ostrings[kk]), U=U), type='gini')
    return ii, jj, state_sparsity, obs_sparsity

def get_parser():
    parser = argparse.ArgumentParser(description="Command line arguments for CSST sparsity sweep")
    parser.add_argument("--dir",     type=str,   default="data",  help="directory to save sparsity data")
    parser.add_argument("--nx",      type=int,   default=2,       help="Number of sites in x direction")
    parser.add_argument("--ny",      type=int,   default=2,       help="Number of sites in y direction")
    parser.add_argument("--nb",      type=int,   default=4,       help="Max Pauli observable weight")
    parser.add_argument("--ham",     type=str,   default='heis',  help="Hamiltonian type",  choices=['tfim','heis'])
    parser.add_argument("--istate",  type=str,   default='+-+-',  help="Initial state ('ghz','w','r','rp','hr','hrp', or length NQ string of [0,1,+,-,>,<])")
    parser.add_argument("--trials",  type=int,   default=10,      help="Number of trials per eps value")
    parser.add_argument("--eps",     type=float, default=0,       help="Reference eps value (kept in the eps sweep, used for labeling)")
    parser.add_argument("--epsmin",  type=float, default=0.0,     help="Minimum eps in the sweep")
    parser.add_argument("--epsmax",  type=float, default=0.3,     help="Maximum eps in the sweep")
    parser.add_argument("--epsnum",  type=int,   default=13,      help="Number of eps values in the sweep")
    parser.add_argument("--gamma",   type=float, default=1e-2,    help="Common decay rate for all collapse operators")
    parser.add_argument("--nw",      type=int,   default=1,       help="Number of workers for multiprocessing")
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    pprint.pprint(vars(args))
    print("\n")

    NX = args.nx
    NY = args.ny
    NQ = NX * NY
    HAM = args.ham
    ISTATE = args.istate
    NB = args.nb
    GAMMA = args.gamma
    EPS = args.eps
    EPSMIN = args.epsmin
    EPSMAX = args.epsmax
    EPSNUM = args.epsnum
    TRIALS = args.trials
    NUM_WORKERS = args.nw

    is_valid_init_state(NQ, ISTATE)

    if EPS:
        LABEL = f"{NX}x{NY}_{HAM}_{ISTATE}_eps={EPS:.1e}"
    else:
        LABEL = f"{NX}x{NY}_{HAM}_{ISTATE}"

    DIR = Path(args.dir) / LABEL
    DIR.mkdir(parents=True, exist_ok=True)

    with open(DIR / "info.json", 'r') as handle:
        info = json.load(handle)

    epss = np.linspace(EPSMIN, EPSMAX, EPSNUM)  # should contain EPS
    assert np.any(np.isclose(epss, EPS)), f"EPS={EPS} not in epss={epss}"

    # import cProfile

    # def profile_body():
    #     NQ, NB, NX, NY, HAM, GAMMA, ISTATE = 5, 4, 1, 5, 'tfim', 1e-2, 'ghz'
    #     U = COB(NQ)
    #     ostrings = pbw(NQ, nb=NB, max=True)
    #     NUM_PAULIS = len(ostrings)
    #     init_state = get_init_state(NQ, ISTATE)

    #     L = get_L(NQ, NX, NY, HAM, eps=0.1, gamma=GAMMA, seed=42)
    #     Lp = U.dag() * L * U
    #     _, V = np.linalg.eig(Lp.full())
    #     Vinv = np.linalg.inv(V)
    #     _ = get_sparsity(expand_into(Vinv=Vinv, state=init_state, U=U), type='gini')
    #     for kk in range(NUM_PAULIS):
    #         _ = get_sparsity(expand_into(V=V, observable=p2op(ostrings[kk]), U=U), type='gini')

    # cProfile.run('profile_body()', sort='cumtime')
    # sys.exit(0)

    mp.set_start_method("spawn", force=True)
    NUM_WORKERS = min(cpu_cap(), NUM_WORKERS)
    print(f"{CYAN}Using multiprocessing with {NUM_WORKERS} workers...{RESET}")
    print("\n")

    tasks = [
        (ii, jj, epss[jj], 42 if ii == 0 else ii)
        for ii in range(TRIALS)
        for jj in range(len(epss))
    ]

    with mp.Pool(processes=NUM_WORKERS,
                 initializer=_init_worker,
                 initargs=(NQ, NX, NY, NB, HAM, GAMMA, ISTATE),
                 ) as pool:
        results = list(tqdm(pool.imap_unordered(_worker, tasks),
                            total=len(tasks),
                            desc="ii,jj sweep",
        ))

    NUM_PAULIS = len(pbw(NQ, nb=NB, max=True))
    state_sparsities = np.zeros((TRIALS, len(epss)), dtype=np.float64)
    observable_sparsities = np.zeros((TRIALS, len(epss), NUM_PAULIS), dtype=np.float64)
    for ii, jj, state_sparsity, obs_sparsity in results:
        state_sparsities[ii, jj] = state_sparsity
        observable_sparsities[ii, jj, :] = obs_sparsity

    np.savez(
        DIR / "sparsities.npz",
        epss=epss,
        state_sparsities=state_sparsities,
        observable_sparsities=observable_sparsities,
    )

