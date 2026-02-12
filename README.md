# CSST
Code and data for CSST paper

## Contents
```
├── ALPHA_GEN.py
├── DATA_ANALYSIS.ipynb
├── DATA_GEN.py
├── environment.yml
├── README.md
└── utils
    ├── cs_utils.py
    ├── pauli_utils.py
    └── shadow_utils.py
```

## Environment
The `environment.yml` specifies all package and dependency versions.  You can recreate this environment by running `conda env create -f environment.yml` in a conda command prompt.  The basic package dependencies are

```
- numpy
- scipy 
- qutip
- scikit-learn 
- matplotlib
- tqdm 
- networkx 
- ipykernel 
- matplotlib-label-lines
- statsmodels
```

## Documentation

### Physical simulation and baseline data generation

Run `DATA_GEN.py` to create the following data files.  The folder `2x2_heis_+-+-` is the default test system, but it is generally set to `{nx}x{ny}_{ham}_{istate}`.

```
└── 2x2_heis_+-+-
    ├── errs.npy
    ├── ests.npy
    ├── exacts.npy
    ├── info.json
    └── shadows.npy
```

| File | Shape | Description |
|----------|----------|----------|
| `exacts.npy`      | `(N, NUM_PAULIS)`         | Exact expectation values over time       |
| `ests.npy`        | `(N, NUM_PAULIS, NSNUM)`  | Shadow estimates                         |
| `errs.npy`        | `(NUM_PAULIS, NSNUM, 8)`  | Error information of shadow estimates        |
| `shadows.npy`     | `(N, NSMAX, NQ, 2)`       | Shadows generated from exact state evolution in QuTiP     |
| `info.json`       |                           | Parameters file of physical simulation and DATA_GEN.py    |

`DATA_GEN.py` takes the following arguments

```
usage: DATA_GEN.py [-h] [--dir DIR] [--nx NX] [--ny NY] [--nb NB] [--ham {tfim,heis,random}]
                   [--istate ISTATE] [--n N] [--nsmin NSMIN] [--nsmax NSMAX] [--nsnum NSNUM] [--nw NW]

Command line arguments for CSST

options:
  -h, --help            show this help message and exit
  --dir DIR             directory to save data files
  --nx NX               Number of sites in x direction
  --ny NY               Number of sites in y direction
  --nb NB               Max Pauli observable weight
  --ham {tfim,heis,random}
                        Hamiltonian type
  --istate ISTATE       Initial state ('ghz','w','r','rp','hr','hrp', or length NQ string of
                        [0,1,+,-,>,<])
  --n N                 Number of time steps
  --nsmin NSMIN         Min number of shadows
  --nsmax NSMAX         Max number of shadows
  --nsnum NSNUM         Number of shadows to try (logspaced)
  --nw NW               Number of workers for multiprocessing
```

The Hamiltonian options are
- `'tfim'`: Transverse-field Ising model $H = J \sum_{\langle a,b\rangle}Z_aZ_b + h \sum_a X_a$
- `'heis'`: Heisenberg model $H = J\sum_{\langle a,b\rangle}\left(X_aX_b+Y_aY_b+Z_aZ_b\right)$

The initial state options are:
- `'ghz'`:  $\text{GHZ}$ state
- `'w'`:    $\text{W}$ state
- `'r'`:    state whose vector elements are uniformly random complex numbers
- `'rp'`:   tensor product of different single-qubit `'r'` states
- `'hr'`:   Haar random state
- `'hrp'`:  tensor product of different single-qubit `'hr'` states
- `[0,1,+,-,>,<]`: a string such as `'0000...'` or `'>+<-...'` indicating which of the 6 Bloch sphere states should be assigned to each qubit
    - `'0'`: $|0\rangle$
    - `'1'`: $|1\rangle$
    - `'+'`: $|+\rangle$
    - `'-'`: $|-\rangle$
    - `'>'`: $|+i\rangle$
    - `'<'`: $|-i\rangle$

### CSST data generation

Run `ALPHA_GEN.py` to create additional data files in the `runs` directory.  Each run of `ALPHA_GEN.py` will create a new pair of `cs_xxxx.npy`, `params_xxxx.json` files.  The index `xxxx` will be the lowest integer unused by any existing pairs.  

```
└── 2x2_heis_+-+-
    ├── errs.npy
    ├── ests.npy
    ├── exacts.npy
    ├── info.json
    ├── runs
    │   ├── cs_0001.npy
    │   └── params_0001.json
    └── shadows.npy
```

| File | Shape | Description |
|----------|----------|----------|
| `/runs/cs_xxxx.npy`      | `(NUM_PAULIS,NSNUM,MNUM,ALPHA_NUM,8)` | CSST error data |
| `/runs/params_xxxx.npy`  |  | Parameters file for alpha sweeps


`ALPHA_GEN.py` takes the following arguments

```
usage: ALPHA_GEN.py [-h] [--dir DIR] [--nx NX] [--ny NY] [--ham {tfim,heis,random}] [--istate ISTATE]
                    [--nw NW] [--mmin MMIN] [--mmax MMAX] [--mnum MNUM] [--amin AMIN] [--amax AMAX]
                    [--anum ANUM] [--snr SNR] [--rescale {0,1}] [--axis {0,1}] [--inverse {0,1}]
                    [--fitint {0,1}] [--replace {0,1}]

Command line arguments for CSST

options:
  -h, --help            show this help message and exit
  --dir DIR             directory to save data files
  --nx NX               Number of sites in x direction
  --ny NY               Number of sites in y direction
  --ham {tfim,heis,random}
                        Hamiltonian type
  --istate ISTATE       Initial state
  --nw NW               Number of workers for multiprocessing
  --mmin MMIN           Min m
  --mmax MMAX           Max m
  --mnum MNUM           Number of ms (logspaced)
  --amin AMIN           Min alpha power (10^amin)
  --amax AMAX           Max alpha power (10^amax)
  --anum ANUM           Number of alphas (logspaced)
  --snr SNR             Filter out (O_i, N_ST) with SNR < value
  --rescale {0,1}       Rescale by sqrt(N/m) in CS reconstruction
  --axis {0,1}          Axis to apply DCT/IDCT to
  --inverse {0,1}       Use IDCT instead of DCT for basis
  --fitint {0,1}        Fit intercept in Lasso regression
  --replace {0,1}       Sample with replacement in time-subsampling
```

The `snr` flag specifies a cutoff signal-to-noise ratio (default is $-1$) which can filter out a large number of iterations so that time isn't wasted reconstructing very noisy signals.  Thus, while `cs_xxxx.npy` can be a large file (GBs) it may contain majority unfilled entries/NaNs depending on the selected cutoff threshold.  This file is read as a `np.memmap` in the plotting code due to its size.

### Plotting

All plots are generated by the `DATA_ANALYSIS.ipynb` notebook.  The data files can be found on Zenodo for the $2 \times 3$ Heisenberg and TFIM models used in the paper.

### Multiprocessing
Some of the data generating functions in `DATA_GEN.py` and `ALPHA_GEN.py` use Python's multiprocessing module to speed things up.  By default 1 worker is requested, but the code will try to use `min(requested cpus, available cpus-1)`.  On a HPC cluster, the number of workers is set to the SLURM environment variable `$SLURM_CPUS_PER_TASK`.

## Funding Acknowledgement
This material is based upon work supported by, or in part by, the U. S. Army Research Laboratory and the U. S. Army Research Office under contract/grant number W911NF2310255.

## Citation
