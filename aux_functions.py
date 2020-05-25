import numpy as np
import libs.matrix as mat

def apply_single_qubit_map(map_func, qubit_index, rho, *args, **kwargs):
    """Applies a single-qubit map to a density matrix of n qubits.

    Args:
        map_func (callable): The map to apply.
            Should be a function that takes a single-qubit density matrix as
            input and applies the map to it.
        qubit_index (int): Index of qubit to apply the map to. 0...n-1
        rho (np.ndarray): Density matrix of n qubits. Shape (2**n, 2**n)
        *args, **kwargs: additional args and kwargs other than rho passed to map_func

    Returns:
        np.ndarray: The density matrix with the map applied. Shape (2**n, 2**n)
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
    return (1 - epsilon) * rho + epsilon * np.dot(np.dot(mat.X, rho), mat.H(mat.X))

def y_noise_channel(rho, epsilon):
    return (1 - epsilon) * rho + epsilon * np.dot(np.dot(mat.Y, rho), mat.H(mat.Y))

def z_noise_channel(rho, epsilon):
    return (1 - epsilon) * rho + epsilon * np.dot(np.dot(mat.Z, rho), mat.H(mat.Z))

def w_noise_channel(rho, alpha):
    return alpha * rho + (1 - alpha) * mat.I(2) / 2
