import os
import numpy as np
import libs.matrix as mat

def binary_entropy(p):
    if p == 1 or p == 0:
        return 0
    else:
        return -p * np.log2(p) - (1 - p) * np.log2((1 - p))

def calculate_keyrate_time(correlations_z, correlations_x, err_corr_ineff, time_interval):
    e_z = 1 - np.sum(correlations_z)/len(correlations_z)
    e_x = 1 - np.sum(correlations_x)/len(correlations_x)
    return len(correlations_z) / time_interval * (1 - binary_entropy(e_x) - err_corr_ineff * binary_entropy(e_z))

def calculate_keyrate_channel_use(correlations_z, correlations_x, err_corr_ineff, resource_list):
    e_z = 1 - np.sum(correlations_z)/len(correlations_z)
    e_x = 1 - np.sum(correlations_x)/len(correlations_x)
    return len(correlations_z) / np.sum(resource_list) * (1 - binary_entropy(e_x) - err_corr_ineff * binary_entropy(e_z))

def assert_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def apply_single_qubit_map(map_func, qubit_index, rho, *args, **kwargs):
    """Applies a single-qubit map to a density matrix of n qubits.

    Parameters
    ----------
    map_func : callable
        The map to apply. Should be a function that takes a single-qubit density
        matrix as input and applies the map to it.
    qubit_index : int
        Index of qubit to which the map is applied. 0...n-1
    rho : np.ndarray
        Density matrix of n qubits. Shape (2**n, 2**n)
    *args, **kwargs: any, optional
        additional args and kwargs other than rho passed to map_func

    Returns
    -------
    np.ndarray
        The density matrix with the map applied. Shape (2**n, 2**n)

    """
    n = int(np.log2(rho.shape[0]))
    rho = rho.reshape((2, 2) * n)
    # there must be a nicer way to do the iteration here:
    out = np.zeros_like(rho)
    for idx in np.ndindex(*(2, 2)*(n-1)):
        my_slice = idx[:qubit_index] + (slice(None),) + idx[qubit_index:n-1+qubit_index] + (slice(None),) + idx[n-1+qubit_index:]
        out[my_slice] = map_func(rho[my_slice], *args, **kwargs)
    return out.reshape((2**n,2**n))

def x_noise_channel(rho, epsilon):
    """A single-qubit bit-flip channel.

    Parameters
    ----------
    rho : np.ndarray
        A single-qubit density matrix (2x2).
    epsilon : scalar
        Error probability 0 <= epsilon <= 1.

    Returns
    -------
    np.ndarray
        The density matrix with the map applied.

    """
    return (1 - epsilon) * rho + epsilon * np.dot(np.dot(mat.X, rho), mat.H(mat.X))

def y_noise_channel(rho, epsilon):
    """A single-qubit bit-and-phase-flip channel.

    Parameters
    ----------
    rho : np.ndarray
        A single-qubit density matrix (2x2).
    epsilon : scalar
        Error probability 0 <= epsilon <= 1.

    Returns
    -------
    np.ndarray
        The density matrix with the map applied.

    """
    return (1 - epsilon) * rho + epsilon * np.dot(np.dot(mat.Y, rho), mat.H(mat.Y))

def z_noise_channel(rho, epsilon):
    """A single-qubit phase-flip channel.

    Parameters
    ----------
    rho : np.ndarray
        A single-qubit density matrix (2x2).
    epsilon : scalar
        Error probability 0 <= epsilon <= 1.

    Returns
    -------
    np.ndarray
        The density matrix with the map applied.

    """
    return (1 - epsilon) * rho + epsilon * np.dot(np.dot(mat.Z, rho), mat.H(mat.Z))

def w_noise_channel(rho, alpha):
    """A single-qubit depolarizing (white) noise channel.

    Parameters
    ----------
    rho : np.ndarray
        A single-qubit density matrix (2x2).
    alpha : scalar
        Error parameter alpha 0 <= alpha <= 1.
        State is fully depolarized with probability (1-alpha)

    Returns
    -------
    np.ndarray
        The density matrix with the map applied.

    """
    return alpha * rho + (1 - alpha) * mat.I(2) / 2 * np.trace(rho) # trace is necessary if dealing with unnormalized states (e.g. in apply_single_qubit_map)
