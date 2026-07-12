import numpy as np
from helpers.gpr import set_params
import matplotlib.pyplot as plt

def lnPost_gp(log_params, gp, k, y, initial_guesses, bounds, prior_std=3):
    '''
    Log-posterior for the GP parameter sampler: Gaussian log-likelihood plus a Gaussian prior
    centred on the MAP estimate with width prior_std in natural-parameter space.
    Returns -inf if any parameter is outside bounds or if the likelihood is non-finite.

    In:
        log_params (array of floats): current GP parameters in log space, length 3k
        gp (celerite2 GaussianProcess): the GP object to evaluate
        k (int): number of SHO terms
        y (array of floats): S-index observations
        initial_guesses (array of floats): log-space MAP parameters used as prior centre
        bounds (ndarray): shape (3k, 2) array of [lower, upper] bounds in log space
        prior_std (float): width of the Gaussian prior in natural-parameter space

    Out:
        log_posterior (float): log-posterior value, or -inf if the point is rejected
    '''
    if np.any(log_params < bounds[:, 0]) or np.any(log_params > bounds[:, 1]):
        return -np.inf
    gp = set_params(log_params, k, gp)
    gp.recompute(quiet=True)
    lnL = gp.log_likelihood(y)
    if not np.isfinite(lnL):
        return -np.inf
    return lnL - 0.5 * np.sum(((np.exp(log_params) - np.exp(initial_guesses)) / prior_std) ** 2)

def plot_trace(sampler, param_names=None):
    '''
    Plots the MCMC walker traces for each parameter in natural (not log) space.

    In:
        sampler (emcee EnsembleSampler): sampler after run_mcmc has been called
        param_names (list of str or None): parameter labels; defaults to param_0, param_1, ...

    Out:
        None
    '''
    samples = sampler.get_chain()  # shape: (n_steps, n_walkers, n_params)
    n_steps, n_walkers, n_params = samples.shape

    if param_names is None:
        param_names = [f"param_{i}" for i in range(n_params)]

    fig, axes = plt.subplots(n_params, 1, figsize=(10, 2.5 * n_params), sharex=True)
    if n_params == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(np.exp(samples[:, :, i]), color="steelblue", alpha=0.3, lw=0.7)
        ax.set_ylabel(param_names[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("Step")
    axes[0].set_title("MCMC Trace")
    fig.tight_layout()
    plt.close(fig)
