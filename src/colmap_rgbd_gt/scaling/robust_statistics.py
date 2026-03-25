"""Robust statistics utilities."""

import numpy as np
from scipy.optimize import minimize


def median_absolute_deviation(data: np.ndarray) -> float:
    median = np.median(data)
    return float(np.median(np.abs(data - median)))


def huber_weights(residuals: np.ndarray, scale: float = 1.345) -> np.ndarray:
    abs_residuals = np.abs(residuals)
    weights = np.ones_like(residuals)
    mask = abs_residuals > scale
    weights[mask] = scale / abs_residuals[mask]
    return weights


def iterative_reweighted_least_squares(
    A: np.ndarray,
    b: np.ndarray,
    max_iter: int = 10,
    tol: float = 1e-6
) -> np.ndarray:
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    for _ in range(max_iter):
        residuals = b - A @ x
        weights = huber_weights(residuals)

        W = np.diag(weights)
        AtWA = A.T @ W @ A
        AtWb = A.T @ W @ b

        x_new = np.linalg.solve(AtWA, AtWb)

        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new

    return x


def remove_outliers_iqr(data: np.ndarray, k: float = 1.5) -> np.ndarray:
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    lower = q1 - k * iqr
    upper = q3 + k * iqr

    return data[(data >= lower) & (data <= upper)]


def remove_outliers_mad(data: np.ndarray, k: float = 3.0) -> np.ndarray:
    median = np.median(data)
    mad = median_absolute_deviation(data)

    if mad == 0:
        return data

    z_scores = 0.6745 * (data - median) / mad
    return data[np.abs(z_scores) < k]


def robust_mean(data: np.ndarray, method: str = "huber") -> float:
    if len(data) == 0:
        return 0.0

    if method == "median":
        return float(np.median(data))
    elif method == "trimmed":
        trimmed = remove_outliers_iqr(data)
        return float(np.mean(trimmed)) if len(trimmed) > 0 else float(np.median(data))
    else:
        return float(np.mean(data))


def robust_std(data: np.ndarray, method: str = "mad") -> float:
    if len(data) == 0:
        return 0.0

    if method == "mad":
        mad = median_absolute_deviation(data)
        return 1.4826 * mad
    else:
        return float(np.std(data))
