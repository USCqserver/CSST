import subprocess, warnings, tqdm

from functools import partial

from sklearn.exceptions import ConvergenceWarning

import numpy as np
from numpy.fft import rfft

import scipy as sp
from scipy.signal import savgol_filter

# import sklearn as skl
from sklearn.linear_model import Lasso

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