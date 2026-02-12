import subprocess, warnings, tqdm

from functools import partial

from sklearn.exceptions import ConvergenceWarning

import numpy as np
from numpy.fft import rfft

import scipy as sp
from scipy.signal import savgol_filter

# import sklearn as skl
from scipy.fftpack import dct, idct
from scipy.sparse import coo_matrix
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from scipy.signal import savgol_filter
from statsmodels.stats.diagnostic import acorr_ljungbox as ljung

from functools import reduce


def basis_pursuit(times, 
                  num_samps, 
                  data, 
                  alpha, 
                  inds=None, 
                  seed=42, 
                  verbose=False, 
                  basis=None,
                  rescale=False,
                  fit_intercept=False,
                  axis=1,
                  inverse=False,
                  replace=False):
    """
    Implements compressed sensing via Basis Pursuit (i.e. Lasso regression) by minimizing
        (1 / (2 * n_samples)) * ||y - Ax||^2_2 + alpha * ||x||_1
    where y is the data, w is y in the sparse basis, and X is the mapping between y and w.  The output is a sparse w
    
    times:      total time array in use
    num_samps:  number of random time points to sample at for CS
    data:       total input data vector
    alpha:      weight of sparsity-penalizing term in lasso problem
    inds:       can provide user-defined inds instead of relying on random sampling
    seed:       random seed for reproducibility
    verbose:    print sparsity of solution
    basis:      user-defined basis, if None use DCT/IDCT
    rescale:    rescale A and y by sqrt(N/m) to preserve energy
    fit_intercept:  whether to fit intercept in Lasso regression
    replace:    whether to sample with replacement when inds is None
    """
 
    if inds is None:    
        rng = np.random.default_rng(seed=seed)
        inds = np.sort(rng.choice(len(times), size=num_samps, replace=replace))
    else:
        if len(inds) != num_samps:
            num_samps = len(inds)
            print(f"Overriding num_samps to {num_samps} to match length of inds")
        assert np.all((inds >= 0) & (inds < len(times)))
        assert len(np.unique(inds)) == len(inds), "All inds must be unique"

    if basis is None:
        # dct(I, axis=0) -> U analysis (rows are basis vectors) wrong
        # idct(I, axis=0) -> U^T synthesis (cols are basis vectors) correct
        # dct(I, axis=1) -> U^T synthesis (cols are basis vectors) correct
        # idct(I, axis=1) -> U analysis (rows are basis vectors) wrong
        if ((axis==0 and inverse is False) or (axis==1 and inverse is True)) and verbose:
            print("Warning: using DCT for synthesis, but axis/inverse settings correspond to analysis")
        if inverse:
            basis = sp.fftpack.idct(np.eye(len(times)), norm='ortho', axis=axis, type=2)
        else:
            basis = sp.fftpack.dct(np.eye(len(times)), norm='ortho', axis=axis, type=2)
    else:
        assert basis.shape[0] == len(times), "Basis vector length != len(times)"

    if verbose:
        is_unitary = np.allclose(basis.conj().T @ basis, np.eye(basis.shape[1]), atol=1e-6)
        if not is_unitary:
            print("Warning: basis is not unitary")

    if rescale:
        scale = np.sqrt(len(times) / num_samps)
    else:
        scale = 1

    # basis = basis[:,1:]

    A = scale * basis[inds]
    y = scale * data[inds].copy()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        lasso = Lasso(alpha=alpha, max_iter=5000, fit_intercept=fit_intercept)
        lasso.fit(X=A, y=y)

    if verbose:
        print_lasso_diagnostics(lasso)

    if fit_intercept:
        cs_result = basis @ lasso.coef_ + lasso.intercept_ / scale
    else:
        cs_result = basis @ lasso.coef_
    return cs_result, lasso

def alpha_sweep(alphas, bp_func, true_signal, eind=2):
    results = []
    for alpha in tqdm.tqdm(alphas):
        recon, _ = bp_func(alpha)
        errors = get_errors(recon, true_signal)
        results.append((alpha, recon, errors[eind]))
    best_ind = np.argmin([r[2] for r in results])
    return results[best_ind], [r[2] for r in results]

def print_lasso_diagnostics(lasso, eps=1e-12):
    """
    Print relevant post-optimization diagnostics for a fitted sklearn Lasso object.
    """
    coef = lasso.coef_
    n_features = coef.size

    n_nonzero = np.count_nonzero(np.abs(coef) > eps)
    sparsity = 1.0 - n_nonzero / n_features

    print("Lasso diagnostics")
    print("-----------------")
    print(f"n_iter_:        {lasso.n_iter_}")
    print(f"dual_gap_:      {lasso.dual_gap_:.3e}")
    print(f"n_features:     {n_features}")
    print(f"n_nonzero:      {n_nonzero}")
    print(f"sparsity:       {100*sparsity:.1f}%")

    if hasattr(lasso, "intercept_"):
        print(f"intercept_:     {lasso.intercept_:.6g}")

    if hasattr(lasso, "alpha"):
        print(f"alpha:          {lasso.alpha}")

def SNR(x, window=10):
    """ Estimate SNR without knowing the noise """
    x = x - x.mean() #not in-place, this creates a new array, so it is safe if x comes from a read-only memmap
    sf = partial(savgol_filter, window_length=window, polyorder=2, axis=0)
    var_smooth = np.var(sf(x))
    var_resid = np.var(x-sf(x))
    return 10 * np.log10(var_smooth/var_resid)

def SNR_exact(signal, noise):
    """Compute the signal-to-noise ratio (SNR) in decibels (dB)."""
    power_signal = np.mean(signal**2)
    power_noise = np.mean(noise**2)
    return 10 * np.log10(power_signal / power_noise) if power_noise != 0 else np.inf

def peak_to_median_ratio_db(x):
    psd = np.abs(rfft(x - np.mean(x)))**2
    med = np.median(psd)
    return 10*np.log10(psd.max() / (med + 1e-16))

def huber_metric(recon, exact, relative=False, delta=None):
    """
    Compute the Huber RMS metric between reconstructed and exact signals.

    Parameters
    ----------
    recon : array_like
        Estimated signal.
    exact : array_like
        Ground-truth signal.
    relative : bool, optional
        If True, compute based on relative residuals (diff / exact).
    delta : float or None, optional
        Threshold for switching between L2 and L1 behavior.
        If None, it's set adaptively as 1.345 * (1.4826 * MAD).

    Returns
    -------
    huber_rms : float
        Root-mean Huber loss (same units as signal if relative=False).
    delta : float
        The threshold actually used.
    """
    recon = np.asarray(recon)
    exact = np.asarray(exact)

    # residuals
    diff = recon - exact
    if relative:
        diff = diff / (np.abs(exact) + 1e-12)

    e = diff.ravel()
    # robust std estimate
    mad = np.median(np.abs(e - np.median(e)))
    sigma_hat = 1.4826 * mad

    if delta is None:
        delta = 1.345 * sigma_hat   # ~95% efficient for Gaussian residuals

    abs_e = np.abs(e)
    quad = abs_e <= delta
    huber_vals = np.empty_like(abs_e)
    huber_vals[quad]  = 0.5 * e[quad]**2
    huber_vals[~quad] = delta * (abs_e[~quad] - 0.5 * delta)

    huber_rms = np.sqrt(np.mean(huber_vals))
    return huber_rms, delta

def get_errors(recon, exact):
    errors = np.zeros(8)

    diff = recon - exact
    rel_diff = diff / (exact + 1e-12)  
    
    # L1 norms
    errors[0] = np.mean(np.abs(diff))           # MAE
    errors[1] = np.mean(np.abs(rel_diff))       # MRE

    #L2 norms
    errors[2] = np.sqrt(np.mean(diff**2))       # RMSE
    errors[3] = np.sqrt(np.mean(rel_diff**2))   # RRMSE

    #Huber norms
    errors[4], _ = huber_metric(recon, exact, relative=False)
    errors[5], _ = huber_metric(recon, exact, relative=True)

    #SNR
    errors[6] = SNR_exact(exact, noise=diff)

    # ljung-box test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        pmin = ljung(recon).lb_pvalue.min()
    errors[7] = pmin
    
    return errors

# ======================================================== #
# ==================== HARMINV UTLITIES =================== #
# ======================================================== #

def plot_fft(times,signals,labels,colors,abs=True,n=None):
    dt = times[1]-times[0]
    if not n:
        n = len(times)
    xs = 2*np.pi*sp.fft.fftshift(sp.fft.fftfreq(n,dt))
    f = plt.figure(figsize=(10,3))
    plt.grid()
    ffts = []
    for signal,label,color in zip(signals,labels,colors):
        # assert len(signal) == N
        ys = np.fft.fftshift(np.fft.fft(a=signal,n=n))
        ffts.append(ys)
        if abs:
            ys = np.abs(ys)
            plt.plot(xs,ys,label=label,color=color)
        else:
            ysre = np.real(ys)
            ysim = np.imag(ys)
            plt.plot(xs,ysre,label=label + " real",color=color)
            plt.plot(xs,ysim,label=label + " imag",color=color,linestyle='--')
    
    plt.legend()
    return f, ffts

def harminv(signal,fmin,fmax,dt,dens=None,nf=None):
    assert not (dens and nf), "Cannot specify both density and number of frequencies"

    def write_data(data):
        with open('temp.txt', 'w') as f:
            for v in data:
                f.write(np.format_float_scientific(v) + '\n')

    path = '../../libraries/harminv-1.4.1/'
    write_data(signal)
    if dens:
        command = f'{path}harminv -t {dt} -d {dens} {fmin}-{fmax} < temp.txt'
    elif nf:
        command = f'{path}harminv -t {dt} -f {nf} {fmin}-{fmax} < temp.txt'
    else:
        command = f'{path}harminv -t {dt} {fmin}-{fmax} < temp.txt'
    output = subprocess.run(command,shell=True,capture_output=True,text=True)
    return output

def get_params_harminv(output):
    """
    Extract the signal parameters from the output of the harminv routine
     - if filter set to True, then extracted components with positive decay rates are not returned
    """
    output = output.stdout.split('\n')[:-1]
    output = output[1:]
    if len(output) != 0:
        harminv = np.array(np.float64([v.split(',') for v in output]))
        harminv = np.delete(harminv, 2, axis=1) #delete the Q factor column
        harminv[:,0] = 2*np.pi*harminv[:,0] #convert frequency to rad/s
        harminv[:,1] = -harminv[:,1] #flip decay rate
        harminv[:,3] = -harminv[:,3] #flip phase
        return harminv
    else:
        raise ValueError("Harminv failed")
    
def print_params(params, fdm=False):
    if fdm:
        titles = ["frequency (rad/s)","decay (1/s)","weights (real)","weights (imag)","error"]
    else:
        titles = ["frequency (rad/s)","decay (1/s)","weights","phases (rad)","error"]

    title = "".join([f"{x:<20}" for x in titles])
    print(title)
    print("-"*len(title))
    for icomp in range(params.shape[0]):
        row = "".join([f"{x:<20.5E}" for x in params[icomp,:]])
        print(row)
    return

def signal(ts,params,sigma=0):
    sig = 0
    for param in params:
        t1,t2,t3,t4 = param[0], param[1], param[2], param[3]
        sig += t3 * np.exp(t2 * ts) * np.cos(t1 * ts + t4)
        # sig += t3 * np.cos(ts * (1j*t1 + t2) + 1j*t4)
    if sigma:
        return sig + np.random.randn(len(ts))*sigma
    else:
        return sig
    
# ======================================================== #
# ==================== MATRIX PENCIL UTLITIES =================== #
# ======================================================== #

def get_params_pencil(output, dt):
    Z, R, M, _ = output            
    inds = np.argsort(np.abs(R))[::-1]
    Z,R = Z[inds], R[inds]
    params_est = np.zeros((len(Z),5),dtype=np.float64)
    params_est[:,0] = np.imag(np.log(Z))/dt
    params_est[:,1] = np.real(np.log(Z))/dt
    params_est[:,2] = np.abs(R)
    params_est[:,3] = np.angle(R)
    params_est[:,4] = np.NaN
    return params_est

def plot_params(data, inds=None, xlim=None, ylim=None):
    """
    data: [true_params, est_params_1, est_params_2, ...]
    inds: which indices of true params are actually sense-able
    """
    if not inds:
        inds = np.arange(len(data[0]))

    markers = ['x','+','o']
    colors = ['C0', 'red', 'blue', 'green', 'purple', 'orange']
    fig = plt.figure(figsize=(10,10))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 6], width_ratios=[6, 1])
    ax = fig.add_subplot(gs[1, 0])
    for ii,params in enumerate(data):
        if ii == 0:
            ax.scatter(params[inds,1],params[inds,0],marker=markers[ii],color=colors[ii])
        else:
            ax.scatter(params[:,1],params[:,0],marker=markers[ii],color=colors[ii])
    ax.grid()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_ylabel(r"$\omega$")
    ax.set_xlabel(r"$\gamma$")

    axt = fig.add_subplot(gs[0, 0], sharex=ax)
    for ii,params in enumerate(data):
        for jj,param in enumerate(params):
            if (ii == 0 and jj in inds) or (ii != 0):
                axt.axvline(x=param[1], color=colors[ii], alpha=0.25, linewidth=1)
    axt.get_yaxis().set_visible(False)  # Hide y-axis labels
    axt.get_xaxis().set_visible(False)  # Hide x-axis labels

    axr = fig.add_subplot(gs[1, 1], sharey=ax)
    for ii,params in enumerate(data):
        for jj,param in enumerate(params):
            if (ii == 0 and jj in inds) or (ii != 0):
                axr.axhline(y=param[0], color=colors[ii], alpha=0.25, linewidth=1)
    axr.get_yaxis().set_visible(False)  # Hide y-axis labels
    axr.get_xaxis().set_visible(False)  # Hide y-axis labels

    plt.subplots_adjust(hspace=0.05, wspace=0.05)

    return fig

def filter_params(params, times, freqtol=None, decaytol=None, amptol=None, matchtol=None, errtol=None):
    '''
    Tolerances used to filter out spurious/unimportant modes
    ftol: 
    amptol:
    decaytol:
    matchtol:
    errtol:
    '''

    N = len(times)
    T = np.max(times)
    dt = times[1] - times[0]
    fNyq = 1/(2*dt)

    def _where_decay_not_too_big(decays, decaytol):
        return np.argwhere(1/np.abs(decays) > decaytol * T).flatten()
    
    def _where_signal_not_constant(freqs, decays, freqtol, decaytol):
        inds = []
        for ii in range(len(freqs)):
            f,d = freqs[ii], decays[ii]
            if 2*np.pi/abs(f) < T/freqtol and 1/abs(d) < T/decaytol:
                inds.append(ii)
        return np.array(inds)

    def _find_matches(freqs, tol):
        indsp = np.argwhere(freqs >= 0)
        indsm = np.argwhere(freqs < 0)
        if len(indsm) == 0:
            return []
        else:
            inds = []
            for ii in indsp:
                mindist = np.min(np.abs(freqs[indsm] + freqs[ii]))
                if mindist < tol:
                    inds.append(ii)
            return inds

    freqs = params[:,0]
    decays = params[:,1]
    amps = params[:,2]
    phases = params[:,3]
    errs = params[:,4]

    #SANITY CHECK
    inds1 = np.argwhere(decays <= 0).flatten() #all decays must be negative
    inds2 = np.argwhere(np.abs(freqs) < 1/(2*dt)).flatten() #all freqs must be less than Nyquist

    #decay can't be too slow AND freq can't be too low, or else the signal is constant
    if freqtol:
        inds3 = _where_signal_not_constant(freqs, decays, freqtol, decaytol)
    else:
        inds3 = np.arange(len(params))

    if decaytol:
        inds4 = _where_decay_not_too_big(decays, decaytol)
    else:
        inds4 = np.arange(len(params))

    #amp must be large enough
    if amptol:
        inds5 = np.argwhere(np.abs(amps) > amptol).flatten()
    else:
        inds5 = np.arange(len(params))

    if matchtol is None:
        inds6 = np.arange(len(params))
    else:
        inds6 = _find_matches(freqs, tol=matchtol)

    try:
        inds7 = np.argwhere(errs <= errtol).flatten()
    except (TypeError, IndexError): #if NaN
        inds7 = np.arange(len(params))
      
    inds = reduce(np.intersect1d, [inds1,inds2,inds3,inds4,inds5,inds6,inds7])

    params2 = np.zeros((len(inds),5),dtype=np.float64)
    params2[:,0] = freqs[inds]
    params2[:,1] = decays[inds]
    params2[:,2] = amps[inds]
    params2[:,3] = phases[inds]
    params2[:,4] = errs[inds]
    #sort by amplitude
    return params2[np.abs(params2[:,2]).argsort()[::-1]] 

'''
@author: zbb
@date: 20190811
@updates:2020-09-17 
@ref: Tapan K. Sakar and Odilon Pereira, Using the Matrix Pencil Method to Estimate the Parameters of Sum of Complex Exponetials, 
IEEE Antennas and Propagation Magazine, Vol. 37, No. 1, February 1995.
'''

def _constructY(y, N, L):
    '''
    y: complex signal sequence.
    N: len(y)
    L: L<N, pencil parameter, N/3 < L < N/2 recommended. 
    return: constructed Y matrix, 
    [
        [y[0], y[1], ..., y[L-1]],
        [y[1], y[1, 0], ..., y[L]], 
        ...
        [y[N-L-1], y[N-L], ..., y[N-1]]
    ]
    (N-L)*(L+1) matrix. 
    '''
    Y = np.zeros((N-L, L+1), dtype=np.complex_)
    for k in range(N-L):
        Y[k, :] = y[k:(k+L+1)]
    return Y 

def _constructZM(Z, N):
    '''
    Z: 1-D complex array.
    return N*M complex matrix (M=len(Z)):
    [
        [1,  1,  1, ..., 1 ],
        [z[0], z[1], .., z[M-1]],
        ...
        [z[0]**(N-1), z[1]**(N-1), ..., z[M-1]**(N-1)]
    ]
    '''
    M = len(Z)
    ZM = np.zeros( (N, M), dtype=np.complex_) 
    for k in range(N):
        ZM[k, :] = Z**k 
    return ZM 

def _SVDFilter(Sp, p=3.0):
    '''
    Sp: 1-D normed eigenvalues of Y after SVD, 1-st the biggest
    p: precise ditigits, default 3.0. 
    return: M, M is the first integer that S[M]/S_max <= 10**(-p)
    '''
    Sm = np.max(Sp) 
    pp = 10.0**(-p)
    for m in range(len(Sp)):
        if Sp[m]/Sm <= pp:
            return m+1 
    return m+1 

def pencil(y, M=None, p=8.0, Lfactor=0.40):
    '''
    Purpose:
      Complex exponential fit of a sampled complex waveform by Matrix Pencil Method.
    Authors: 
      Zbb
    Arguments:
      N    - number of data samples. ==len(y)       [INPUT]
      y    - 1-D complex array of sampled signal.   [INPUT]
      dt   - sample interval.                       [INPUT]
      M    - pencil parameter. 
             if None: use p to determin M.
             if given in range(0, Lfractor*N), then use it
             if given out of range, then use p to determin M.
      p    - precise digits of signal, default 8.0, corresponding to 10**(-8.0).
    Returns: (Z, R, M, (residuals, rank, s))
      Z    - 1-D Z array. 
      R    - 1-D R array.
      M    - M in use. 
      (residuals, rank, s)   - np.linalg.lstsq further results. 
    Method:
      y[k] = y(k*dt) = sum{i=0--> M} R[i]*( Z[i]**k ) 
      Z[i] = exp(si*dt)
    
    Comment: 
      To some extent, it is a kind of PCA method. 
    '''
    N = len(y)
    # better between N/3~N/2, pencil parameter:
    L = int(N*Lfactor)  
    # construct Y matrix (Hankel data matrix) from signal y[i], shape=(N-L, L+1):
    Y = _constructY(y, N, L)
    # SVD of Y: 
    _, S, V = np.linalg.svd(Y, full_matrices=True)
    #results: U.shape=(N-L, N-L), S.shape=(L+1, ), V.shape=(L+1, L+1)

    # find M: 
    if M is None:
        M = _SVDFilter(np.abs(S), p=p)
    elif M not in range(0, L+1):
        M = _SVDFilter(np.abs(S), p=p) 
    else: 
        pass
    
    # matrix primes based on M:
    #Vprime = V[0:M, :] # remove [M:] data set. only 0, 1, 2, ..., M-1 remains
    #Sprime = S[0:M]
    V1prime = V[0:M, 0:-1] # remove last column
    V2prime = V[0:M, 1:] # remove first column
    #smat = np.zeros((U.shape[0], M), dtype=np.complex_)
    #smat[:M, :M] = np.diag(Sprime)
    #Y1 = np.dot(U[:-1, :], np.dot(smat, V1prime))

    V1prime_H_MPinv = np.linalg.pinv(V1prime.T) # find V1'^+ , Moore-Penrose pseudoinverse 
    V1V2 = np.dot(V1prime_H_MPinv, V2prime.T) # construct V1V2 = np.dot(V1'^+, V2') 
    Z = np.linalg.eigvals(V1V2) # find eigenvalues of V1V2. Zs.shape=(M,)
    #print(V1V2.shape, Z)

    # find R by solving least-square problem: Y = np.dot(ZM, R)
    ZM = np.row_stack([Z**k for k in range(N)]) # N*M 
    R, residuals, rank, s = np.linalg.lstsq(ZM, y, rcond=-1)
    return (Z, R, M, (residuals, rank, s))
    
