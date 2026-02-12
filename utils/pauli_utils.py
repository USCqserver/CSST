import itertools,random,math
import matplotlib.pyplot as plt

from typing import Sequence, Union

import numpy as np
import scipy as sp
import qutip as qt
import networkx as nx

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

# ============================================================= #
# ==================== PAULI UTILITIES =================== #
# ============================================================= #

def get_inds(nq,ptype=None):
    """
    In the Pauli basis ordered by weight, return the indices that partition the array into blocks
    of equal Pauli weight

    each index is the first element of some weight block

    If ptype is specified, then instead a 2D array is return, where the first axis is the weight block,
    and the second axis are the actual indices in that weight block which 
    """
    Ps = pbw(nq)
    temp = [list(group) for key, group in itertools.groupby(Ps, key=p2w)]
    inds = [[list(Ps).index(p) for p in arr if is_type(p, ptype)] for arr in temp]
    return inds

def num_paulis(nq, nb=None, maxw=False):
    """ Returns the number of Pauli strings on nq qubits of (max) weight nb analytically """
    if nb is None:
        return 4**nq
    elif maxw:
        return sum([math.comb(nq,k)*(3**k) for k in range(1,nb+1)])
    else:
        return math.comb(nq,nb)*(3**nb)

def is_type(P,types=None):
    """ Check if Pauli string P is made up of only those operators in types """
    if types:
        return set(P).issubset({*types,'I'})
    else:
        return True

def p2w(P):
    """ Returns the weight of a Pauli string """
    nq = len(P)
    return nq - P.count('I')

def psort(Ps):
    """ Sort Pauli strings by weight and then lexicographically """
    return sorted(Ps, key=lambda s: (p2w(s), s))

def p2op(op_string, norm=False):
    """
    convert string of operators containing I, X, Y, Z to QuTiP operator
    string can be of form: ['I','X','X',...] or 'IXX...' or ('I','X','X',...)

    If dim is specified, return the normalize Pauli string P/sqrt(dim)
    """
    assert type(op_string) == str or type(op_string) == list or type(op_string) == tuple
    if not norm:
        return qt.tensor([PAULI_STR_TO_PAULI[p] for p in op_string])
    else:
        return qt.tensor([PAULI_STR_TO_PAULI[p] for p in op_string])/np.sqrt(2**len(op_string))

def ps2op(ops,norm=False):
    return sum(c * p2op(p,norm=norm) for (c,p) in ops)

def perturb(ops, eps=0, seed=42):
    rng = np.random.default_rng(seed)
    return [(c * (1 + eps * rng.standard_normal()), p) for (c,p) in ops]

def pbw(nq,nb=-1,ptype=None,max=False,k=None,edges=None):
    """
    Generate all possible Pauli strings of weight nb acting on nq qubits.
    If nb=-1, all possible Pauli strings of weight 0 to nq are generated.
    If ptype is specified, only Pauli strings of that type are generated.
    If max is True, all Pauli strings of weight 0 to nb are generated.
    If edges is specified, only Pauli strings that act on the specified interacting qubits are generated.
        edges = [... (i,j) ...] where i,j are qubit indices
    If k is specified, k Pauli strings are randomly selected.
    """

    def _helper(nq,nb):
        Ps = []
        for loc in itertools.combinations(range(nq),nb): #locations of error
            P = ['I']*nq
            for err in list(itertools.product(['X','Y','Z'],repeat=nb)): #actual error
                for ii in range(len(loc)):
                    P[loc[ii]] = err[ii]
                Ps.append(''.join(P))
        return Ps

    assert nb==-1 or 0 <= nb <= nq

    if nb == -1:
        Ps = list(itertools.product(['I','X','Y','Z'],repeat=nq))
        Ps = [''.join(O) for O in Ps]
    else:
        if max:
            Ps = []
            for w in range(0,nb+1):
                Ps += _helper(nq,w)
        else:
            Ps = _helper(nq,nb)

    if ptype:
        assert ptype in ['X','Y','Z']
        Ps = list(filter(lambda x: is_type(x, ptype),Ps))

    def _in_edges(P,edges):
        spots = np.argwhere(np.array(list(P)) != 'I').flatten()
        for interaction in edges:
            if set(spots) == set(interaction):
                return True
        return False
    
    if edges:
        Ps = list(filter(lambda x: _in_edges(x,edges),Ps))

    #should always go last
    if k:
        Ps = random.sample(Ps,k=k)

    return psort(Ps)

def rand_op(Ps,bound=1,herm=True,seed=42):
    rng = np.random.default_rng(seed)
    coeffs = rng.uniform(low=-bound,high=bound,size=len(Ps))
    if not herm:
        coeffs = np.array(coeffs, dtype=complex)
        coeffs += 1j*rng.uniform(low=-bound,high=bound,size=len(coeffs))
        coeffs = coeffs/2
    return list(zip(coeffs, Ps))

def amp_damp_ops(nq):
    Xs = pbw(nq,nb=1,ptype='X')
    Ys = pbw(nq,nb=1,ptype='Y')
    return [(p2op(a)+1j*p2op(b))/2 for (a,b) in zip(Xs,Ys)]

# ============================================================= #
# ==================== MODEL UTILITIES =================== #
# ============================================================= #

def heisenberg_model_ops(nq, edges, jx=1, jy=1, jz=1, eps=0):
   ops = []
   for (ii,jj) in edges:
        for jval,p in zip([jx, jy, jz], ['X', 'Y', 'Z']):
            op = ['I']*nq
            op[ii] = p
            op[jj] = p
            op = ''.join(op)
            ops.append((jval, op))
   return ops

def tfim_ops(nq, edges, two_body='Z', one_body='X', j2b=1, j1b=1):
    ops = []
    for (ii,jj) in edges:
        op = ['I']*nq
        op[ii] = two_body
        op[jj] = two_body
        ops.append((j2b, ''.join(op)))
    for ii in range(nq):
        op = ['I']*nq
        op[ii] = one_body
        ops.append((j1b, ''.join(op)))
    return ops

def get_eigval_bounds_loose(H_ops, all_gammas, ub):
    """Compute upper bounds on the magnitude of the real and imag parts of the eigenvalues of the Liouvillian
    This assumes that each Pauli is unnormalized, i.e. has spectral norm 1.
    H_ops: list of (coeff, pauli_string) tuples defining the Hamiltonian
    all_gammas: list of dissipation rates for each collapse operator
    ub: upper bound on spectral norm of any collapse operator
    """
    C = np.sum(np.abs([coeff for (coeff,pauli) in H_ops]))
    B = np.sum(all_gammas) * ub**2
    return 2*B, 2*B + 2*C

def get_eigval_bounds_tight(H_ops, c_ops):
    """ warning: will build the full dense operators """
    H = ps2op(H_ops)
    LHnorm = np.linalg.norm(qt.liouvillian(H, c_ops=[]).full(), ord=2)
    LDnorm = np.linalg.norm(qt.liouvillian(0*H, c_ops).full(), ord=2)
    return LDnorm, LHnorm + LDnorm

def get_h_ops(nq,model,graph=None,seed=None,eps=0,k=None):
    if graph:
        assert nq == graph.number_of_nodes(), "# of graph nodes must match nq"
        edges = graph.edges
    else:
        edges = None

    if model == 'tfim':
        assert edges, "Must provide graph for tfim model"
        H_ops = tfim_ops(nq, edges=edges, j1b=-3, j2b=-1)
    elif model == 'heis':
        assert edges, "Must provide graph for heisenberg model"
        H_ops = heisenberg_model_ops(nq, edges=edges, jx=1, jy=1, jz=1)
    elif model == 'random':
        assert seed, "Must provide seed for random model"
        temp = pbw(nq, edges=edges, k=k)
        H_ops = rand_op(temp, seed=seed)
    else:
        raise ValueError(f"Unknown model '{model}'")

    H_ops = perturb(H_ops, eps=eps)
    return H_ops

def get_init_state(
    nq: int,
    spec: Union[str, qt.Qobj],
    *,
    seed: int = 42,
) -> qt.Qobj:
    
    char_lookup = {
        '0': k0,
        '1': k1,
        '+': kp,
        '-': km,
        '>': kpi,
        '<': kmi,
    }

    # --- Dispatch on type of `spec` ---
    if isinstance(spec, str):
        name = spec.strip().lower()
        if name == 'ghz':
            return qt.ghz_state(nq)
        elif name == 'w':
            dim = 2**nq
            state = np.zeros(dim, dtype=complex)
            for k in range(nq):
                index = 1 << (nq - k - 1)  # position of '1' in binary string
                state[index] = 1
            return qt.Qobj(state / np.sqrt(nq), dims=[[2]*nq, [1]*nq])
        elif name == 'r':
            state = qt.rand_ket(2**nq, seed=seed, distribution='fill')
            state.dims = [[2]*nq, [1]*nq]
            return state
        elif name == 'hr':
            state = qt.rand_ket(2**nq, seed=seed, distribution='haar')
            state.dims = [[2]*nq, [1]*nq]
            return state
        elif name == 'rp':
            kets = [qt.rand_ket(2, seed=seed + i, distribution='fill') for i in range(nq)]
            return qt.tensor(kets)
        elif name == 'hrp':
            kets = [qt.rand_ket(2, seed=seed + i, distribution='haar') for i in range(nq)]
            return qt.tensor(kets)
        else:
            # Treat as compact per-qubit string (ignore whitespace)
            compact = "".join(ch for ch in spec if not ch.isspace())
            if len(compact) != nq:
                raise ValueError(f"Product-state string length {len(compact)} != nq={nq}.")
            try:
                kets = [char_lookup[ch] for ch in compact]
            except KeyError as e:
                allowed = "".join(char_lookup.keys())
                raise ValueError(
                    f"Invalid character {e.args[0]!r} in product-state string. "
                    f"Allowed: {allowed}."
                )
            return qt.tensor(kets)

    # if isinstance(spec, qt.Qobj):
    #     H = spec
    #     evals, evecs = H.eigenstates()
    #     if eigvals < 1 or eigvals > len(evecs):
    #         raise ValueError(f"eigvals must be in [1, {len(evecs)}], got {eigvals}.")
    #     state = evecs[0] if eigvals == 1 else sum(evecs[:eigvals])
    #     state = state.unit()
    #     state.dims = [[2]*nq, [1]*nq]
    #     return state

    raise TypeError(
        "Unsupported `spec` type. Use: str (named state or compact product string) "
        "or qutip.Qobj (Hamiltonian)."
    )

