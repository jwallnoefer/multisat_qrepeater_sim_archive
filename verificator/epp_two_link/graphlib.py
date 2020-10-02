# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 14:33:42 2015
@author: julius
The aim of this project is to provide functions for Pauli-diagonal
noise channels and the ADB multipartite entanglement purification protocol for
two-colorable graph states. (Ref. PhysRevLett.91.107903, PhysRevA.71.012319)
All states are assumed to be diagonal in the graph state basis corresponding to
the graph state considered.
The noise should work for arbitrary graph states, while the purification
protocol only makes sense for two-colorable graph states.
Make sure the variables input follow the conventions given in the docstrings,
since not many sanity checks are included.
This should run reasonably well even for bigger states (ca. 10 qubits) if
cythonized.
"""

import numpy as np
from math import log


def adj_matrix(N, E):
    '''Returns the adjacency matrix for a graph with `N` vertices
    (labeled 0 to N-1) and the edges `E`.
    Parameters
    ----------
    N : int
        Number of vertices.
    E : list (or tuple) of tuples
        Should contain 2-tuples with the edges of the graph. Each pair
        (i,j) indicates a connection between vertices i and j. Only
        simple, unweighted, undirected graphs are supported.
    Returns
    -------
    adj : numpy.ndarray
        Adjacency matrix of the graph specified. Is a symmetric `N` x `N` matrix
        with N_{ij}=1 if (i,j) is in `E` and 0 otherwise.
    '''
    adj = np.zeros((N, N), dtype=int)
    for i in range(N):
        for n in range(N):
            if (i, n) in E:
                adj[i, n] = 1
                adj[n, i] = 1
    return adj


class graph:
    def __init__(self, N, E, set_a=[], set_b=[], set_c=[], set_d=[]):
        self.N = N
        self.a = set_a
        self.b = set_b
        self.c = set_c
        self.d = set_d
        self.E = E
        self.adj = adj_matrix(N, E)


def noisy(rho, nn, graph=0):
    '''Template to generate noise patterns.
    In physical terms this is correlated sigma_z noise on all particles in `nn`.
    Parameters
    ----------
    rho : numpy.ndarray
        Is the state acted on. Should be a 2**N-dimensional vector
        with the diagonal entries of the density matrix in the graph
        state basis.
    nn : list of int
        The list of which qubits are affected, counting starts at 0.
        Indices are expected to be in order
        012...(N-1) regardless of coloring of the vertices.
    adj : graph, optional
        Specifies the graphstate considered.
        This function does not use it - only for consistency.
    Returns
    -------
    out : numpy.ndarray
        The state after the action. Same shape as `rho`.
    '''
    N = int(log(len(rho), 2))
    rho = rho.reshape((2,) * N)
    for n in nn:
        rho = np.swapaxes(np.swapaxes(rho, 0, n)[::-1], 0, n)
    rho = rho.reshape((2**N,))
    return rho

#    #old, slow implementation
#    j=0
#    for n in nn:
#        k=int(log(len(rho),2))-1-n
#        j=j^(1<<k) # + would be slightly faster than ^ but can lead to weird behaviour
#    mu=np.zeros(len(rho))
#    for i in range(len(mu)):
#        mu[i]=rho[i^j]
#    return mu


def znoisy(rho, n, graph=0):
    '''Applies sigma_z noise on the specified qubit.
    Parameters
    ----------
    rho : numpy.ndarray
        Is the state acted on. Should be a 2**N-dimensional vector
        with the diagonal entries of the density matrix in the graph
        state basis.
    n : int
        The `n`-th qubit is affected, counting starts at 0. Indices are
        expected to be in order
        012...(N-1) regardless of coloring of the vertices.
    graph : graph, optional
        Specifies the graphstate considered.
        This function does not use it. Included only so znoisy can be called
        the same way as xnoisy and ynoisy.
    Returns
    -------
    out : numpy.ndarray
        The state after the action. Same shape as `rho`.
    '''
    return noisy(rho, [n])


def xnoisy(rho, n, graph):
    '''Applies sigma_x noise on the specified qubit.
    Parameters
    ----------
    rho : numpy.ndarray
        Is the state acted on. Should be a 2**N-dimensional vector
        with the diagonal entries of the density matrix in the graph
        state basis.
    n : int
        The `n`-th qubit is affected, counting starts at 0. Indices are
        expected to be in order
        012...(N-1) regardless of coloring of the vertices.
    graph : graph
        Specifies the graphstate considered.
    Returns
    -------
    out : numpy.ndarray
        The state after the action. Same shape as `rho`.
    '''
    nn = []
    for i in range(graph.N):
        if graph.adj[n, i]:
            nn += [i]
    return noisy(rho, nn)


def ynoisy(rho, n, graph):
    '''Applies sigma_y noise on the specified qubit.
    Parameters
    ----------
    rho : numpy.ndarray
        Is the state acted on. Should be a 2**N-dimensional vector
        with the diagonal entries of the density matrix in the graph
        state basis.
    n : int
        The `n`-th qubit is affected, counting starts at 0. Indices are
        expected to be in order
        012...(N-1) regardless of coloring of the vertices.
    graph : graph
        Specifies the graphstate considered.
    Returns
    -------
    out : numpy.ndarray
        The state after the action. Same shape as `rho`.
    '''
    nn = [n]
    for i in range(graph.N):
        if graph.adj[n, i]:
            nn += [i]
    return noisy(rho, nn)


def znoise(rho, n, p, graph=0):
    return p * rho + (1 - p) * znoisy(rho, n, graph)


def xnoise(rho, n, p, graph):
    return p * rho + (1 - p) * xnoisy(rho, n, graph)


def ynoise(rho, n, p, graph):
    return p * rho + (1 - p) * ynoisy(rho, n, graph)


def wnoise(rho, n, p, graph):
    return p * rho + (1 - p) / 4 * (rho + xnoisy(rho, n, graph) + ynoisy(rho, n, graph) + znoisy(rho, n, graph))


def noise_pattern(rho, n, ps, graph):
    '''Applies a local pauli-diagonal noise channel on the specified qubit
    Parameters
    ----------
    rho : numpy.ndarray
        Is the state acted on. Should be a 2**N-dimensional vector
        with the diagonal entries of the density matrix in the graph
        state basis.
    n : int
        The `n`-th qubit is affected, counting starts at 0. Indices are
        expected to be in order
        012...(N-1) regardless of coloring of the vertices.
    ps : list
        The coefficients of the noise channel.
        Should have 4 entries p_0 p_x p_y p_z .
    graph : graph
        Specifies the graphstate considered.
    Returns
    -------
    out : numpy.ndarray
        The state after the action. Same shape as `rho`.
    '''
    return ps[0] * rho + ps[1] * xnoisy(rho, n, graph) + ps[2] * ynoisy(rho, n, graph) + ps[3] * znoisy(rho, n, graph)


def wnoise_all(rho, p, graph):
    for i in range(int(log(len(rho), 2))):
        rho = wnoise(rho, i, p, graph)
    return rho


def noise_global(rho, p, graph=0):
    k = len(rho)
    return p * rho + (1 - p) * np.ones(k) / k


# fidelity for diagonal entries
def fidelity(rho, mu):
    a = np.sqrt(rho)
    b = np.sqrt(mu)
    return np.dot(a, b)


# trace distance for diagonal entries
def trdist(rho, mu):
    sigma = np.abs(rho - mu)
    return np.sum(sigma)


# normalize the state to trace = 1,
# also catches numerical phenomena with entries < 0
def normalize(rho):
    mu = np.copy(rho)
    if np.any(mu < 0):
        mu[mu < 0] = 0
        mu = normalize(mu)
    K = np.sum(mu)
    return mu / K


def mask_a(j, graph):
    l = 0
    for k in range(len(graph.a)):
        if j & (1 << k):
            l = l ^ (1 << (graph.N - 1 - graph.a[k]))
    return l


def mask_b(j, graph):
    l = 0
    for k in range(len(graph.b)):
        if j & (1 << k):
            l = l ^ (1 << (graph.N - 1 - graph.b[k]))
    return l


# Note: np.fromfunction does not help with speeding up p1p2 - cythonizing does!
def p1(rho, graph):
    mu = np.zeros(len(rho))
    for i in range(2**graph.N):
        j = i & (mask_b((1 << len(graph.b)) - 1, graph))
        for k in range(2**len(graph.b)):
            l = mask_b(k, graph)
            mu[i] += rho[(i ^ j) ^ l] * rho[i ^ l]
    mu = normalize(mu)
    return mu


def p2(rho, graph):
    mu = np.zeros(len(rho))
    for i in range(2**graph.N):
        j = i & (mask_a((1 << len(graph.a)) - 1, graph))
        for k in range(2**len(graph.a)):
            l = mask_a(k, graph)
            mu[i] += rho[(i ^ j) ^ l] * rho[i ^ l]
    mu = normalize(mu)
    return mu


def p1_var(rho, sigma, graph):
    mu = np.zeros(len(rho))
    for i in range(2**graph.N):
        j = i & (mask_b((1 << len(graph.b)) - 1, graph))
        for k in range(2**len(graph.b)):
            l = mask_b(k, graph)
            mu[i] += rho[(i ^ j) ^ l] * sigma[i ^ l]
    mu = normalize(mu)
    return mu


def p2_var(rho, sigma, graph):
    mu = np.zeros(len(rho))
    for i in range(2**graph.N):
        j = i & (mask_a((1 << len(graph.a)) - 1, graph))
        for k in range(2**len(graph.a)):
            l = mask_a(k, graph)
            mu[i] += rho[(i ^ j) ^ l] * sigma[i ^ l]
    mu = normalize(mu)
    return mu


def mask_k(j, graph, myset):
    l = 0
    for k in range(len(myset)):
        if j & (1 << k):
            l = l ^ (1 << (graph.N - 1 - myset[k]))
    return l


def pk(rho, sigma, graph1, graph2, myset):
    mu = np.zeros(len(rho))
    notmyset = [i for i in range(graph1.N) if i not in myset]
    for i in range(2**graph1.N):
        j = i & (mask_k((1 << len(notmyset)) - 1, graph1, notmyset))
        for k in range(2**len(notmyset)):
            l = mask_k(k, graph1, notmyset)
            mu[i] += rho[(i ^ j) ^ l] * sigma[i ^ l]
    mu = normalize(mu)
    return mu
