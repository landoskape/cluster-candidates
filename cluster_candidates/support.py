import math
import numpy as np
import numba as nb
from scipy.signal import correlate


try:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_available = True

except ImportError:
    device = "cpu"
    torch_available = False


@nb.njit
def condensed_to_square_indices(k, n):
    """
    Convert a condensed matrix index k to square matrix indices (i, j).

    Parameters:
    k : int
        Index in the condensed matrix
    n : int
        Number of points in the original data

    Returns:
    (i, j) : tuple of ints
        Indices in the square distance matrix
    """
    # First solve for i
    i = n - 2 - math.floor(math.sqrt(-8 * k + 4 * n * (n - 1) - 7) / 2.0 - 0.5)
    # Then solve for j
    j = k + i + 1 - n * (n - 1) // 2 + (n - i) * ((n - i) - 1) // 2

    if i >= n or j >= n or i < 0 or j < 0:
        raise ValueError("Indices out of bounds")

    return i, j


@nb.njit
def square_to_condensed_index(i, j, n):
    """
    Convert square matrix indices (i, j) to a condensed matrix index.

    Parameters:
    i, j : int
        Indices in the square distance matrix where i < j
    n : int
        Number of points in the original data

    Returns:
    k : int
        Index in the condensed matrix
    """
    if i >= n or j >= n or i < 0 or j < 0:
        raise ValueError("Indices out of bounds")

    if i < j:
        k = (n * (n - 1) // 2) - (n - i) * ((n - i) - 1) // 2 + j - i - 1
    else:
        k = (n * (n - 1) // 2) - (n - j) * ((n - j) - 1) // 2 + i - j - 1
    return k


@nb.njit(parallel=True, fastmath=True)
def dist_between_points(xpoints, ypoints):
    N = len(xpoints)
    max_k = square_to_condensed_index(N - 1, N - 1, N) + 1
    dists = np.zeros(max_k)
    for k in nb.prange(max_k):
        i, j = condensed_to_square_indices(k, N)
        dists[k] = np.sqrt((xpoints[i] - xpoints[j]) ** 2 + (ypoints[i] - ypoints[j]) ** 2)
    return dists


@nb.njit(parallel=True)
def pair_val_from_vec(vector):
    N = len(vector)
    max_k = square_to_condensed_index(N - 1, N - 1, N) + 1
    value1 = np.zeros(max_k)
    value2 = np.zeros(max_k)
    for k in nb.prange(max_k):
        i, j = condensed_to_square_indices(k, N)
        value1[k] = vector[i]
        value2[k] = vector[j]
    return value1, value2


@torch.no_grad()
def _torch_corrcoef(data: np.ndarray, undefined_val: float = np.nan) -> np.ndarray:
    """
    Compute the correlation matrix of the input data using PyTorch on the GPU,
    then return back the numpy array on the CPU.

    Only use when your data matrix is huge!
    """
    data_gpu = torch.tensor(data).to(device)
    torch_corr = torch.corrcoef(data_gpu)
    if ~np.isnan(undefined_val):
        no_variance = torch.var(data_gpu, dim=1) == 0
        idx_no_variance = torch.where(no_variance)[0]
        torch_corr.scatter_(
            dim=0,
            index=idx_no_variance.view(-1, 1).expand(-1, torch_corr.size(1)),
            value=undefined_val,
        )
        torch_corr.scatter_(
            dim=1,
            index=idx_no_variance.view(1, -1).expand(torch_corr.size(0), -1),
            value=undefined_val,
        )
    torch_corr = torch_corr.to("cpu").numpy()
    return torch_corr


def _numpy_corrcoef(data: np.ndarray, undefined_val: float = np.nan) -> np.ndarray:
    """
    Compute the correlation matrix using NumPy.
    This is a fallback when PyTorch is not available.
    """
    corr = np.corrcoef(data)
    if ~np.isnan(undefined_val):
        no_variance = np.var(data, axis=1) == 0
        idx_no_variance = np.where(no_variance)[0]
        corr[idx_no_variance, :] = undefined_val
        corr[:, idx_no_variance] = undefined_val
    return corr


def corrcoef(data: np.ndarray, undefined_val: float = np.nan) -> np.ndarray:
    """
    Compute the correlation matrix using PyTorch if available, otherwise fall back to NumPy.

    Parameters:
    -----------
    data : np.ndarray
        Input data matrix
    undefined_val : float, optional
        Value to use for undefined correlations (default: np.nan)

    Returns:
    --------
    np.ndarray
        Correlation matrix
    """
    if torch_available:
        return _torch_corrcoef(data, undefined_val)
    else:
        return _numpy_corrcoef(data, undefined_val)


def compute_cross_correlations(activity_matrix, max_lag=None, normalize=True):
    """
    Compute cross-correlation for all pairs of neurons at different lags.

    Parameters:
    - activity_matrix: 2D numpy array of shape (n_neurons, n_timepoints),
                       where each row is the activity of a neuron over time.
    - max_lag: Maximum lag to compute cross-correlation for. If None, uses full length.
    - normalize: Whether to normalize the cross-correlation by signal energy.

    Returns:
    - lags: 1D numpy array of lag values.
    - cross_corrs: 3D numpy array of shape (n_neurons, n_neurons, num_lags),
                   where cross_corrs[i, j, :] gives the cross-correlation
                   between neuron i and neuron j at different lags.
    """
    n_neurons, n_timepoints = activity_matrix.shape
    if max_lag is None:
        max_lag = n_timepoints - 1  # Full range of possible lags

    lags = np.arange(-max_lag, max_lag + 1)
    cross_corrs = np.zeros((n_neurons, n_neurons, len(lags)))

    for i in range(n_neurons):
        for j in range(i, n_neurons):  # Compute only upper triangle, use symmetry
            corr = correlate(activity_matrix[i], activity_matrix[j], mode="full", method="auto")
            mid = len(corr) // 2
            corr = corr[mid - max_lag : mid + max_lag + 1]  # Trim to desired lags

            if normalize:
                norm_factor = np.sqrt(np.sum(activity_matrix[i] ** 2) * np.sum(activity_matrix[j] ** 2))
                if norm_factor > 0:
                    corr /= norm_factor

            cross_corrs[i, j, :] = corr
            cross_corrs[j, i, :] = corr  # Exploit symmetry

    return lags, cross_corrs
