"""
The maps that model the different processes in the QKD return for input that is diagonal in Bell-basis a diagonal output.
To reduce calculations I determined in the scipt "How many numbers for state" the effect of the maps on the diagonal elements
"""
import numpy as np
import functools

"""These are some helper functions. a-d represents the diagonal elements of the first state, e-h the ones of the second state"""

z_rot = lambda a, b, c, d: np.array([b, a, d, c])
y_rot = lambda a, b, c, d: np.array([d, c, b, a])

perf_dist = lambda a, b, c, d, e, f, g, h: np.array(
    [a * e, d * h, a * g, d * f, d * e, a * h, d * g, a * f, c * g, b * f, c * e, b * h, b * g, c * f, b * e, c * h])
dc0 = lambda ae, af, ag, ah, be, bf, bg, bh, ce, cf, cg, ch, de, df, dg, dh: np.array(
    [ae + af, be + bf, ce + cf, de + df])
dc1 = lambda ae, af, ag, ah, be, bf, bg, bh, ce, cf, cg, ch, de, df, dg, dh: np.array(
    [ae + af + ag + ah, be + bf + bg + bh, ce + cf + cg + ch, de + df + dg + dh])
"""p is the ideality of the map, q = 1-p"""
mixnswap = lambda p, q, a, b, c, d, e, f, g, h: np.array([a * e * p + b * f * p + c * g * p + d * h * p + q / 4,
                                                          a * f * p + b * e * p + c * h * p + d * g * p + q / 4,
                                                          a * g * p + b * h * p + c * e * p + d * f * p + q / 4,
                                                          a * h * p + b * g * p + c * f * p + d * e * p + q / 4])


def dp_sing(t, T, a, b, c, d):
    """ Calculate the state after dephasing for one memory for time t.
    Parameters
    ----------
    t : float 
        time of dephasig
    T : float
            dephasing time of the memory
    a-d: float
            diagonal elements of the state

    Returns
    -------
    list of diagonal elements of the state after dephasing

    """
    lam = (1 - np.exp(-t / (2 * T))) / 2
    return ((1 - lam) * np.array([a, b, c, d]) + lam * z_rot(a, b, c, d)).tolist()


def dp_doub(t, T, a, b, c, d):
    """ Calculate the state after dephasing for time t1 for one memory and t2 for the other memory.
    Parameters
    ----------
    t : float 
        time of dephasig
    T : float
            dephasing time of the memories
    a-d: float
            diagonal elements of the state

    Returns
    -------
    list of diagonal elements of the state after dephasing

    """
    lam = (1 - np.exp(- t / (2 * T))) / 2
    lam = lam + lam - 2 * lam**2
    return ((1 - lam) * np.array([a, b, c, d]) + lam * z_rot(a, b, c, d)).tolist()


def coupl(em, a, b, c, d):
    """ Calculate the state after imperfect coupling to the fibre.
    Parameters
    ----------
    em1, em2 : float 
        misalignment errors of the stations (0-1)
    a-d: float
            diagonal elements of the state

    Returns
    -------
    list of diagonal element of the state after coupling

    """
    p = 1 - em
    q = em
    return (p * np.array([a, b, c, d]) + q * y_rot(a, b, c, d)).tolist()


@functools.lru_cache(maxsize=2048)
def distil(lam, pd1, pd2, a, b, c, d, e, f, g, h):
    """ Calculate the state after imperfect entanglement distillation and dephasing.
    Parameters
    ----------
    lam1, lam2 : float
            idealities of the distillation process of the stations
    pd1, pd2 : float
            probabilities for dark counts in the measurement for the stations
    a-d: float
            diagonal elements of the fist state
    e-h: float
            diagonal elements of the second state

    Returns
    -------
    list of diagonal element of the state after dephasing, probability for acceptance of the distillation result

    """
    p0 = (1 - pd1) * (1 - pd2)  # probability for zero dark counts
    # probability for one or two dark counts
    p1 = 0.5 * (pd1 + pd2 - pd1 * pd2)
    mixed = (lam * perf_dist(a, b, c, d, e, f, g, h) + (1 - lam) * np.ones((16)) /
             16).tolist()  # mixing the result of the perfect map with abs mixed state
    # state times the accapance probability
    unnormed = p0 * dc0(*mixed) + p1 * dc1(*mixed)
    trace = np.sum(unnormed)  # acceptance probability
    normed = (unnormed / trace).tolist()  # normalising the state
    return normed, trace


def swap(lam, a, b, c, d, e, f, g, h):
    """ Calculate the state after imperfect entanglement swapping and dephasing.
    Parameters
    ----------
    lam: float
            idealities of the swapping process of the middle station
    a-d: float
            diagonal elements of the fist state
    e-h: float
            diagonal elements of the second state

    Returns
    -------
    list of diagonal element of the state after swapping

    """
    swapped = mixnswap(lam, 1 - lam, a, b, c, d, e, f, g, h)
    normed = swapped / np.sum(swapped)  # normalising the state
    return normed
