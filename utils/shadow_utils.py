from tqdm import tqdm
import qutip as qt
import numpy as np

I,X,Y,Z = qt.qeye(2),qt.sigmax(),qt.sigmay(),qt.sigmaz()
Had, S = qt.Qobj([[1,1],[1,-1]])/np.sqrt(2), qt.Qobj([[1,0],[0,1j]])
k0b1 = qt.projection(2,0,1)
k1b0 = qt.projection(2,1,0)
k0,k1 = qt.basis(2,0),qt.basis(2,1)
kp,km,kpi,kmi = Had*k0,Had*k1,S*Had*k0,S*Had*k1

PAULI_STR_TO_IDX = {'I':0, 'X':1, 'Y':2, 'Z':3}
IDX_TO_PAULI_STR = np.array(['I','X','Y','Z'])
PAULI_STR_TO_PAULI = {'I':I, 'X':X, 'Y':Y, 'Z':Z}
PAULI_STR_TO_U = {'X':Had,'Y':Had*S.dag(),'Z':I}

def num_shadows(epsilon, delta, M, ostrings):
    """
    Computes an upper bound on the number of shadows necessary to estimate all obersvables in ostrings up to error=error_tol and 
    with success probability=succ_prob, assuming the use of median-of-means estimation

    NOTE: this assumes that the operators are Pauli strings
    """
    kloc = max([len([x for x in ostr if x!='I']) for ostr in ostrings])     #max locality
    N = (34/(epsilon**2))*(4**kloc)*(1)  #for now observables are all Paulis so max eigval is 1
    K = 2*np.log(2*M/delta)
    return N, K, kloc

def apply_U_to_rho_tensor(rho_tensor, basis_idx):
    """
    apply tensor product operator efficiently to density matrix in tensor form
    """
    T = rho_tensor
    nq = len(basis_idx)
    for qind, idx in enumerate(basis_idx):
        U = PAULI_STR_TO_U[IDX_TO_PAULI_STR[idx]].full()  # single-qubit unitary to rotate into measurement basis

        # Apply on ket axis qind
        T = np.tensordot(U, T, axes=([1], [qind]))     # new axis 0 is U's row
        T = np.moveaxis(T, 0, qind)
        # Apply on bra axis nq+qind
        T = np.tensordot(U.conj(), T, axes=([1], [nq + qind]))
        T = np.moveaxis(T, 0, nq + qind)
    return T

def get_shadows(state, NUM_SHADOWS, derandom=False, seed=None):
    """
    Obtain NS classical shadows of an N-qubit state.
    """
    rng = np.random.default_rng(seed)

    if derandom == True:
        raise NotImplementedError("Derandomized protocol not yet implemented")

    nq = len(state.dims[0])
    d = 2**nq

    rho_tensor = state.full().reshape([2]*2*nq)  # density matrix in tensor form
    shadows = np.empty((NUM_SHADOWS, nq, 2), dtype=np.int8)  #each shadow is a list of (basis_idx, outcome) pairs
    for ii in range(NUM_SHADOWS):
        basis_idx = rng.integers(1, 4, size=nq, dtype=np.uint8)   #measurement bases for this shadow
        shadows[ii,:,0] = basis_idx
        if state.isket:
            pdist = np.array(np.abs(state.full().flatten())**2, dtype=np.float64)
        else:
            snapshot = apply_U_to_rho_tensor(rho_tensor, basis_idx).reshape((d, d))
            pdist = np.array(np.real(np.diag(snapshot)), dtype=np.float64)
        # U = qt.tensor([PAULI_STR_TO_U[pauli] for pauli in IDX_TO_PAULI_STR[basis_idx]])  #unitary to rotate into measurement basis
        # snapshot = U * state * U.dag()
        pdist[pdist < 1e-15] = 0.0
        sample = rng.choice(d, p=pdist/pdist.sum())
        sample = np.array(list(f"{sample:0{nq}b}"), dtype=np.int8)
        shadows[ii,:,1] = 1 - 2 * sample  # 0 -> +1, 1 -> -1

        # free big temporaries promptly
        # del U, snapshot, pdist

    return shadows

def estimate_batch(shadows, ostrings, shadow_subs):
    NUM_SHADOWS, _, _ = shadows.shape
    NUM_PAULIS = len(ostrings)
    NSS = len(shadow_subs)

    ests = np.zeros((NUM_PAULIS, NSS))
    encoded_ostrings = encode_ostrings(ostrings)
    
    order = np.arange(NUM_SHADOWS)  # no shuffling needed since shadows are already random
    L = 0
    running_sum = np.zeros(NUM_PAULIS, dtype=float)
    running_cnt = np.zeros(NUM_PAULIS, dtype=int)
    for ii, M in enumerate(shadow_subs):
        new_idx = order[L:M]  # the next block
        batch = np.array([shadows[jj] for jj in new_idx])
        shots = estimate_exp(batch, encoded_ostrings) #has shape (M-L, NUM_PAULIS)

        running_sum += shots.sum(axis=0)
        running_cnt += M-L

        ests[:, ii] = safe_mean(running_sum, running_cnt)
        L = M
    return ests

def encode_ostrings(ostrings):
    # convert ['XIZY...', ...] to [[1,0,3,2,...],...]
    nq = len(ostrings[0])
    M = len(ostrings)
    temp = np.empty((M, nq), dtype=np.uint8)
    for ii, Ostring in enumerate(ostrings):
        for jj, pauli in enumerate(Ostring):
            temp[ii, jj] = PAULI_STR_TO_IDX[pauli]
    return temp

def safe_mean(sum_arr, cnt_arr, fill_value=0):
    # returns sum/cnt with fill_value where cnt==0
    return np.divide(sum_arr, cnt_arr,
                     out=np.full(sum_arr.shape, fill_value, dtype=float),
                     where=cnt_arr != 0)

def estimate_exp(shadows, eostrings):
    """
    Use shadows to estimate each Ostring
    shadows: (# shadows, # qubits, 2) array of (basis, outcome)
    eostrings: (# observables, # qubits) array of encoded Pauli strings
    """
    bases  = shadows[..., 0].astype(np.uint8)
    outcomes = shadows[..., 1].astype(np.int8)

    # Broadcast to compare bases with observables:
    # bases → (# shadows,1,# qubits), obs → (1,# observables,# qubits)
    bases3d = bases[:, None, :]
    obs3d   = eostrings[None, :, :]

    # mask of which qubits matter (non-I)
    mask = obs3d != 0        
    weights = mask[0].sum(axis=1)  # Pauli weights (# observables,)  
    scales = (3.0 ** weights).astype(np.float64)

    # matches[i,j] = does shadow i contribute to obs j on all indices where obs j is not I?
    matches = np.all((bases3d == obs3d) | (~mask), axis=2)  

    # vals_eff[i,j,k] = outcome of shadow i on qubit k of obs j, if obs j on index k is not I, else 1
    vals_eff = np.where(mask, outcomes[:, None, :], 1)     

    # products[i,j] = product of vals_eff[i,j,:] over k
    products = np.prod(vals_eff, axis=2, dtype=np.int8)   

    # get pauli weight array and convert to 3**w
    w = mask[0].sum(axis=1)                     # (M,)
    scale = (3.0 ** w).astype(np.float64)       # (M,)

    # per-shot contributions: 3^w * product if match else 0
    # valid = np.where(matches, products.astype(np.int64), 0) 
    shots = np.where(matches, products, 0).astype(np.float64) * scale  # (S,M)
    return shots