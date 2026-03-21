"""Comprehensive evaluation metrics for probabilistic downscaling.

Metrics:
  - CRPS (Continuous Ranked Probability Score)
  - Per-variable RMSE and MAE
  - Spread-skill ratio
  - Rank histogram (ensemble calibration)
  - Power spectra (radially averaged, Hann-windowed)
  - Q-Q quantiles
"""

import numpy as np
from typing import Optional


def crps_ensemble(observations: np.ndarray, ensemble: np.ndarray) -> float:
    """Compute CRPS for an ensemble forecast.

    CRPS = E|X - y| - 0.5 * E|X - X'|

    Args:
        observations: (N,) array of observations
        ensemble: (M, N) array of M ensemble members, N locations

    Returns:
        Scalar CRPS averaged over all locations.
    """
    M, N = ensemble.shape
    # E|X - y|: mean absolute error across ensemble
    mae = np.mean(np.abs(ensemble - observations[None, :]), axis=0)  # (N,)

    # E|X - X'|: mean pairwise absolute difference
    # Efficient: sort ensemble, use order statistics
    sorted_ens = np.sort(ensemble, axis=0)  # (M, N)
    # For sorted values, E|X-X'| = 2/(M^2) * sum_{i=1}^{M} (2i - M - 1) * x_{(i)}
    weights = 2 * np.arange(1, M + 1) - M - 1  # (M,)
    pairwise = np.sum(weights[:, None] * sorted_ens, axis=0) / (M * M)  # (N,)

    crps = mae - pairwise  # (N,)
    return float(np.mean(crps))


def rmse(prediction: np.ndarray, target: np.ndarray,
         mask: Optional[np.ndarray] = None) -> float:
    """Root mean squared error, optionally masked (e.g., land-only)."""
    diff = prediction - target
    if mask is not None:
        diff = diff[mask]
    return float(np.sqrt(np.mean(diff ** 2)))


def mae(prediction: np.ndarray, target: np.ndarray,
        mask: Optional[np.ndarray] = None) -> float:
    """Mean absolute error, optionally masked."""
    diff = np.abs(prediction - target)
    if mask is not None:
        diff = diff[mask]
    return float(np.mean(diff))


def spread_skill_ratio(ensemble: np.ndarray, target: np.ndarray) -> float:
    """Ratio of ensemble spread to ensemble mean RMSE.

    SSR = 1.0 means well-calibrated.
    SSR < 1.0 means under-dispersive (overconfident).
    SSR > 1.0 means over-dispersive.

    Args:
        ensemble: (M, ...) ensemble predictions
        target: (...) observation
    """
    ens_mean = ensemble.mean(axis=0)
    spread = ensemble.std(axis=0).mean()
    skill = np.sqrt(np.mean((ens_mean - target) ** 2))
    return float(spread / max(skill, 1e-10))


def rank_histogram(ensemble: np.ndarray, observation: np.ndarray,
                   num_bins: Optional[int] = None) -> np.ndarray:
    """Compute rank histogram for ensemble calibration.

    For each observation, find its rank among ensemble members.
    A well-calibrated ensemble produces a uniform histogram.

    Args:
        ensemble: (M, N) ensemble predictions
        observation: (N,) observations
        num_bins: number of bins (default: M + 1)

    Returns:
        Histogram counts of shape (num_bins,).
    """
    M, N = ensemble.shape
    if num_bins is None:
        num_bins = M + 1

    # For each location, count how many ensemble members are below observation
    ranks = np.sum(ensemble < observation[None, :], axis=0)  # (N,)
    hist, _ = np.histogram(ranks, bins=num_bins, range=(0, M))
    return hist


def qq_quantiles(prediction: np.ndarray, target: np.ndarray,
                 n_quantiles: int = 100) -> tuple:
    """Compute quantile-quantile values.

    Args:
        prediction: predicted values (flattened)
        target: observed values (flattened)
        n_quantiles: number of quantile points

    Returns:
        (pred_quantiles, target_quantiles) arrays of shape (n_quantiles,)
    """
    probs = np.linspace(0, 1, n_quantiles)
    pred_q = np.quantile(prediction.flatten(), probs)
    target_q = np.quantile(target.flatten(), probs)
    return pred_q, target_q


def power_spectrum_2d(field: np.ndarray, dx_km: float = 4.0,
                      apply_hann: bool = True) -> tuple:
    """Compute radially averaged power spectrum with Hann windowing.

    Args:
        field: 2D field (H, W)
        dx_km: grid spacing in km
        apply_hann: apply 2D Hann window before FFT

    Returns:
        (wavelengths_km, power) arrays
    """
    H, W = field.shape

    if apply_hann:
        hann_y = np.hanning(H)
        hann_x = np.hanning(W)
        window = np.outer(hann_y, hann_x)
        field = field * window

    # 2D FFT
    fft2 = np.fft.fft2(field)
    power2d = np.abs(fft2) ** 2 / (H * W)

    # Radial averaging
    ky = np.fft.fftfreq(H, d=dx_km)
    kx = np.fft.fftfreq(W, d=dx_km)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)

    k_max = min(1 / (2 * dx_km), K.max())
    k_bins = np.linspace(0, k_max, min(H, W) // 2)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])

    power = np.zeros(len(k_centers))
    for i in range(len(k_centers)):
        mask = (K >= k_bins[i]) & (K < k_bins[i + 1])
        if mask.sum() > 0:
            power[i] = power2d[mask].mean()

    wavelengths = 1.0 / np.maximum(k_centers, 1e-10)  # km
    return wavelengths, power


def per_variable_metrics(ensemble: np.ndarray, target: np.ndarray,
                         var_names: list) -> dict:
    """Compute CRPS, RMSE, MAE for each variable.

    Args:
        ensemble: (M, N_vars, H, W) ensemble predictions
        target: (N_vars, H, W) observation
        var_names: list of variable names

    Returns:
        Dict mapping var_name -> {crps, rmse, mae, spread_skill}
    """
    results = {}
    M = ensemble.shape[0]

    for i, var in enumerate(var_names):
        ens_var = ensemble[:, i].reshape(M, -1)  # (M, H*W)
        tgt_var = target[i].flatten()  # (H*W,)
        mask = ~np.isnan(tgt_var)
        ens_var = ens_var[:, mask]
        tgt_var = tgt_var[mask]

        results[var] = {
            "crps": crps_ensemble(tgt_var, ens_var),
            "rmse": rmse(ens_var.mean(axis=0), tgt_var),
            "mae": mae(ens_var.mean(axis=0), tgt_var),
            "spread_skill": spread_skill_ratio(ens_var, tgt_var),
        }

    return results
