import os
import numpy as np
import libs.matrix as mat


sqrt_plus_ix = 1 / np.sqrt(2) * (mat.I(2) + 1j * mat.X)
sqrt_minus_ix = mat.H(sqrt_plus_ix)
bilateral_cnot = np.dot(mat.CNOT(0, 2, N=4), mat.CNOT(1, 3, N=4))
dejmps_operator = np.dot(bilateral_cnot, mat.tensor(sqrt_minus_ix, sqrt_plus_ix, sqrt_minus_ix, sqrt_plus_ix))
dejmps_operator_dagger = mat.H(dejmps_operator)
dejmps_proj_ket_z0z0 = mat.tensor(mat.I(4), mat.z0, mat.z0)
dejmps_proj_ket_z1z1 = mat.tensor(mat.I(4), mat.z1, mat.z1)
dejmps_proj_bra_z0z0 = mat.H(dejmps_proj_ket_z0z0)
dejmps_proj_bra_z1z1 = mat.H(dejmps_proj_ket_z1z1)


def binary_entropy(p):
    if p == 1 or p == 0:
        return 0
    else:
        return -p * np.log2(p) - (1 - p) * np.log2((1 - p))


def calculate_keyrate_time(correlations_z, correlations_x, err_corr_ineff, time_interval):
    e_z = 1 - np.sum(correlations_z) / len(correlations_z)
    e_x = 1 - np.sum(correlations_x) / len(correlations_x)
    return len(correlations_z) / time_interval * (1 - binary_entropy(e_x) - err_corr_ineff * binary_entropy(e_z))


def calculate_keyrate_channel_use(correlations_z, correlations_x, err_corr_ineff, resource_list):
    e_z = 1 - np.sum(correlations_z) / len(correlations_z)
    e_x = 1 - np.sum(correlations_x) / len(correlations_x)
    return len(correlations_z) / np.sum(resource_list) * (1 - binary_entropy(e_x) - err_corr_ineff * binary_entropy(e_z))


def assert_dir(path):
    """Check if `path` exists, and create it if it doesn't.

    Parameters
    ----------
    path : str
        The path to be checked/created.

    Returns
    -------
    None

    """
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
    for idx in np.ndindex(*(2, 2) * (n - 1)):
        my_slice = idx[:qubit_index] + (slice(None),) + idx[qubit_index:n - 1 + qubit_index] + (slice(None),) + idx[n - 1 + qubit_index:]
        out[my_slice] = map_func(rho[my_slice], *args, **kwargs)
    return out.reshape((2**n, 2**n))


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
    return alpha * rho + (1 - alpha) * mat.I(2) / 2 * np.trace(rho)  # trace is necessary if dealing with unnormalized states (e.g. in apply_single_qubit_map)


def dejmps_protocol(rho):
    """Applies the DEJMPS entanglement purification protocol.

    Input is usually two entangled pairs and output is one entangled pair if
    successful.
    This protocol was introduced in:
    D. Deutsch, et. al., Phys. Rev. Lett., vol. 77, pp. 2818–2821 (1996)
    arXiv:quant-ph/9604039


    Parameters
    ----------
    rho : np.ndarray
        Four-qubit density matrix (16x16).

    Returns
    -------
    p_suc : scalar
        probability of success for the protocol
    state : np.ndarray
        Two-qubit density matrix (4x4). The state of the remaining pair IF the
        protocol was successful.
    """
    rho = np.dot(np.dot(dejmps_operator, rho), dejmps_operator_dagger)
    rho = np.dot(np.dot(dejmps_proj_bra_z0z0, rho), dejmps_proj_ket_z0z0) + np.dot(np.dot(dejmps_proj_bra_z1z1, rho), dejmps_proj_ket_z1z1)
    p_suc = np.trace(rho)
    state = rho / p_suc  # renormalize
    return p_suc, state
