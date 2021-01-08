import os
import numpy as np
import libs.matrix as mat
import pandas as pd


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


def calculate_keyrate_time(correlations_z, correlations_x, err_corr_ineff, time_interval, return_std=False):
    e_z = 1 - np.mean(correlations_z)
    e_x = 1 - np.mean(correlations_x)
    pair_per_time = len(correlations_z) / time_interval
    keyrate = pair_per_time * (1 - binary_entropy(e_x) - err_corr_ineff * binary_entropy(e_z))
    if not return_std:
        return keyrate
    # use error propagation formula
    keyrate_std = pair_per_time * np.sqrt((-np.log2(e_x) + np.log2(1 - e_x))**2 * np.std(correlations_x)
                                          + err_corr_ineff**2 * (-np.log2(e_z) + np.log2(1 - e_z))**2 * np.std(correlations_x)
                                          )
    return keyrate, keyrate_std


def calculate_keyrate_channel_use(correlations_z, correlations_x, err_corr_ineff, resource_list, return_std=False):
    e_z = 1 - np.mean(correlations_z)
    e_x = 1 - np.mean(correlations_x)
    pair_per_resource = len(correlations_z) / np.sum(resource_list)
    keyrate = pair_per_resource * (1 - binary_entropy(e_x) - err_corr_ineff * binary_entropy(e_z))
    if not return_std:
        return keyrate
    # use error propagation formula
    keyrate_std = pair_per_resource * np.sqrt((-np.log2(e_x) + np.log2(1 - e_x))**2 * np.std(correlations_x)
                                              + err_corr_ineff**2 * (-np.log2(e_z) + np.log2(1 - e_z))**2 * np.std(correlations_x)
                                              )
    return keyrate, keyrate_std


def standard_bipartite_evaluation(data_frame, err_corr_ineff=1):
    states = data_frame["state"]

    fidelity_list = np.real_if_close([np.dot(np.dot(mat.H(mat.phiplus), state), mat.phiplus)[0, 0] for state in states])
    fidelity = np.mean(fidelity_list)
    fidelity_std = np.std(fidelity_list)

    z0z0 = mat.tensor(mat.z0, mat.z0)
    z1z1 = mat.tensor(mat.z1, mat.z1)
    correlations_z = np.real_if_close([np.dot(np.dot(mat.H(z0z0), state), z0z0)[0, 0] + np.dot(np.dot(mat.H(z1z1), state), z1z1)[0, 0] for state in states])

    x0x0 = mat.tensor(mat.x0, mat.x0)
    x1x1 = mat.tensor(mat.x1, mat.x1)
    correlations_x = np.real_if_close([np.dot(np.dot(mat.H(x0x0), state), x0x0)[0, 0] + np.dot(np.dot(mat.H(x1x1), state), x1x1)[0, 0] for state in states])

    key_per_time, key_per_time_std = calculate_keyrate_time(correlations_z=correlations_z,
                                                            correlations_x=correlations_x,
                                                            err_corr_ineff=err_corr_ineff,
                                                            time_interval=data_frame["time"].iloc[-1],
                                                            return_std=True)
    key_per_resource, key_per_resource_std = calculate_keyrate_channel_use(correlations_z=correlations_z,
                                                                           correlations_x=correlations_x,
                                                                           err_corr_ineff=err_corr_ineff,
                                                                           resource_list=data_frame["resource_cost_max"],
                                                                           return_std=True)
    return [fidelity, fidelity_std, key_per_time, key_per_time_std, key_per_resource, key_per_resource_std]


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
        additional args and kwargs passed to map_func

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


def apply_m_qubit_map(map_func, qubit_indices, rho, *args, **kwargs):
    """Applies an m-qubit map to a density matrix of n qubits.

    Parameters
    ----------
    map_func : callable
        The map to apply. Should be a function that takes a single-qubit density
        matrix as input and applies the map to it.
    qubit_indices : list of ints
        Indices of qubit to which the map is applied. Indices from 0...n-1
    rho : np.ndarray
        Density matrix of n qubits. Shape (2**n, 2**n)
    *args, **kwargs: any, optional
        additional args and kwargs passed to map_func

    Returns
    -------
    np.ndarray
        The density matrix with the map applied. Shape (2**n, 2**n)

    """
    m = len(qubit_indices)
    # if m == 1:
    #     return apply_single_qubit_map(map_func=map_func, qubit_index=qubit_indices[0], rho=rho, *args, **kwargs)
    n = int(np.log2(rho.shape[0]))
    rho = rho.reshape((2, 2) * n)
    assert m <= n
    qubit_indices = sorted(qubit_indices)
    index_list = qubit_indices + [n + qubit_index for qubit_index in qubit_indices]
    # still not found a nicer way for the iteration here
    out = np.zeros_like(rho)
    for idx in np.ndindex(*(2, 2) * (n - m)):
        my_slice = list(idx)
        for current_idx in index_list:
            my_slice.insert(current_idx, slice(None))
        my_slice = tuple(my_slice)
        # print(idx, n, m, qubit_indices, index_list)
        # print(my_slice)
        out[my_slice] = map_func(rho[my_slice].reshape(2**m, 2**m), *args, **kwargs).reshape((2, 2) * m)
    return out.reshape((2**n, 2**n))

# def apply_m_qubit_map_alternate(map_func, qubit_indices, rho, *args, **kwargs):
#     m = len(qubit_indices)
#     n = int(np.log2(rho.shape[0]))
#     rho = rho.reshape((2, 2) * n)
#     assert m <= n
#     qubit_indices = sorted(qubit_indices)
#     index_list = qubit_indices + [n + qubit_index for qubit_index in qubit_indices]
#     perm_list = [i for i in range(2 * n)]
#     unperm_list = [i for i in range(2 * (n - m))]
#     for j, current_idx in enumerate(index_list):
#         perm_list.remove(current_idx)
#         perm_list += [current_idx]
#         unperm_list.insert(current_idx, 2 * (n - m) + j)
#     rho = rho.transpose(perm_list).reshape((2, 2) * (n - m) + (2**m, 2**m))
#     map_func = np.vectorize(map_func, signature="(i,j)->(i,j)")
#     out = map_func(rho).reshape((2, 2) * n)
#     # print(n, m, qubit_indices, index_list)
#     # print(perm_list, unperm_list)
#     return out.transpose(unperm_list).reshape((2**n, 2**n))


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
