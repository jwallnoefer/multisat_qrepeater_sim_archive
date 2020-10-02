# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 14:59:42 2015
@author: julius
Most of the funcitons should be properly updated now.
"""

import numpy as np
import graphlib as gg
from math import log
from cmath import sqrt


# Defining elementary vectors and operators - never modify these in a function!!
z0 = np.array([1, 0]).reshape(2, 1)
z1 = np.array([0, 1]).reshape(2, 1)
x0 = 1 / sqrt(2) * np.array([1, 1]).reshape(2, 1)
x1 = 1 / sqrt(2) * np.array([1, -1]).reshape(2, 1)
y0 = 1 / sqrt(2) * np.array([1, 1j]).reshape(2, 1)
y1 = 1 / sqrt(2) * np.array([1, -1j]).reshape(2, 1)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]])
Ha = 1 / sqrt(2) * np.array([[1, 1], [1, -1]])  # Hadamard gate


def H(rho):  # Hermitian Conjugate - because only matrices are allowed to matrix.H
    return rho.conj().T


Pz0 = np.dot(z0, H(z0))
Pz1 = np.dot(z1, H(z1))
Px0 = np.dot(x0, H(x0))
Px1 = np.dot(x1, H(x1))
Py0 = np.dot(y0, H(y0))
Py1 = np.dot(y1, H(y1))


def I(n):
    return np.eye(n, dtype=complex)  # so we can call the Identity matrix with I(n)


def tensor(*args):
    '''
    Returns the matrix representation of the tensor product of an arbitrary
    number of matrices.
    '''
    res = np.array([[1]])
    for i in args:
        res = np.tensordot(res, i, ([], [])).swapaxes(1, 2).reshape(res.shape[0] * i.shape[0], res.shape[1] * i.shape[1])
    return res


phiplus = 1 / sqrt(2) * (tensor(z0, z0) + tensor(z1, z1))
phiminus = 1 / sqrt(2) * (tensor(z0, z0) - tensor(z1, z1))
psiplus = 1 / sqrt(2) * (tensor(z0, z1) + tensor(z1, z0))
psiminus = 1 / sqrt(2) * (tensor(z0, z1) - tensor(z1, z0))


def znoisy(rho, n):  # znoise on the nth qubit, start counting from 0
    N = int(log(rho.shape[0], 2))
    noise = np.array([[1]])
    for i in range(N):
        if i == n:
            noise = tensor(noise, Z)
        else:
            noise = tensor(noise, I(2))
    return np.dot(np.dot(noise, rho), noise)


def xnoisy(rho, n):
    N = int(log(rho.shape[0], 2))
    noise = np.array([[1]])
    for i in range(N):
        if i == n:
            noise = tensor(noise, X)
        else:
            noise = tensor(noise, I(2))
    return np.dot(np.dot(noise, rho), noise)


def ynoisy(rho, n):
    N = int(log(rho.shape[0], 2))
    noise = np.array([[1]])
    for i in range(N):
        if i == n:
            noise = tensor(noise, Y)
        else:
            noise = tensor(noise, I(2))
    return np.dot(np.dot(noise, rho), noise)


def znoise(rho, n, p):
    return p * rho + (1 - p) * znoisy(rho, n)


def xnoise(rho, n, p):
    return p * rho + (1 - p) * xnoisy(rho, n)


def ynoise(rho, n, p):
    return p * rho + (1 - p) * ynoisy(rho, n)


def wnoise(rho, n, p):
    return p * rho + (1 - p) / 4 * (rho + xnoisy(rho, n) + ynoisy(rho, n) + znoisy(rho, n))


def wnoise_all(rho, p):
    N = int(log(rho.shape[0], 2))
    mu = np.copy(rho)
    for n in range(N):
        mu = wnoise(mu, n, p)
    return mu


def noise_global(rho, p):
    return p * rho + (1 - p) * I(rho.shape[0])


def CNOT(n, m, N):
    '''gives the N-qubit CNOT unitary with with the n-th qubit as source
    and the m-th qubit as target
    '''
    if n == m:
        raise ValueError('Nonsensical Input: CNOT acts on two qubits')
    a = np.array([[1]])
    b = np.array([[1]])
    for i in range(N):
        if i == n:
            a = tensor(a, Pz0)
            b = tensor(b, Pz1)
        elif i == m:
            a = tensor(a, I(2))
            b = tensor(b, X)
        else:
            a = tensor(a, I(2))
            b = tensor(b, I(2))
    return a + b


def CZ(n, m, N):
    '''gives the N-qubit CZ unitary acting on n-th and m-th qubit'''
    # construct unitary
    if n == m:
        raise ValueError('Nonsensical Input: CZ acts on two qubits')
    a = np.array([[1]])
    b = np.array([[1]])
    for i in range(N):
        if i == n:
            a = tensor(a, Pz0)
            b = tensor(b, Pz1)
        elif i == m:
            a = tensor(a, I(2))
            b = tensor(b, Z)
        else:
            a = tensor(a, I(2))
            b = tensor(b, I(2))
    return a + b


def Ucnot(psi, n, m):
    # works both with (n,) and (n,1) vectors
    nn = int(log(psi.shape[0], 2))
    return np.dot(CNOT(n, m, nn), psi)


def Ucz(psi, n, m):
    # works both with (n,) and (n,1) vectors
    nn = int(log(psi.shape[0], 2))
    return np.dot(CZ(n, m, nn), psi)


def Mcnot(rho, n, m):
    nn = int(log(rho.shape[0], 2))
    unitary = CNOT(n, m, nn)
    return np.dot(np.dot(unitary, rho), H(unitary))


def Mcz(rho, n, m):
    nn = int(log(rho.shape[0], 2))
    unitary = CZ(n, m, nn)
    return np.dot(np.dot(unitary, rho), H(unitary))


def graphstate(graph):
    aux = [x0] * graph.N
    psi = tensor(*aux)
    for edge in graph.E:
        psi = Ucz(psi, *edge)
    return psi


def complement_op(n, graph):
    '''Gives the operator associated with the local complement at vertex n
    '''
    a = np.array([[1]])
    for i in range(graph.N):
        if i == n:
            a = tensor(a, 1 / sqrt(2) * (I(2) - 1j * X))
        elif graph.adj[i, n]:
            a = tensor(a, 1 / sqrt(2) * (I(2) + 1j * Z))
        else:
            a = tensor(a, I(2))
    return a


def complement_graph(n, graph):  # not sure what this is doing in the matrix module
    '''Returns the new graph after the local complement at vertex n
    '''
    # breaks if not a simple graph
    # surely this can be done better

    # get neighbourhood of n
    Nn = []
    for i in range(graph.N):
        if graph.adj[i, n]:
            Nn += [i]
    new_adjmatrix = np.copy(graph.adj)
    for i in Nn:
        for k in Nn:
            if k == i:
                continue
            new_adjmatrix[i, k] = (graph.adj[i, k] + 1) % 2
    # get new edges from adjmatrix
    new_edges = []
    for i in range(graph.N):
        for k in range(i + 1, graph.N):
            if new_adjmatrix[i, k]:
                new_edges += [(i, k)]
    return gg.graph(graph.N, new_edges)  # deletes sets a to d


def Ugraph(rho, graph):
    '''
    Transform a matrix given in the computational basis
    to the graph state basis.
    '''
    # again here the specific form of the graph state enters
    gstate = graphstate(graph)
    mytuple = ()
    for i in range(2**graph.N):  # get all 2**N basis states
        operator = np.array([[1]])
        for n in range(graph.N):  # build operator to generate state
            if i & (1 << ((graph.N - 1) - n)):
                operator = tensor(operator, Z)
            else:
                operator = tensor(operator, I(2))
        mytuple += (np.dot(operator, gstate),)
    U = H(np.hstack(mytuple))
    return np.dot(np.dot(U, rho), H(U))


def Ungraph(rho, graph):
    '''
    Transform a matrix given in the graph state basis
    to the computational basis.
    '''
    # again here the specific form of the graph state enters
    gstate = graphstate(graph)
    mytuple = ()
    for i in range(2**graph.N):  # get all 16 basis states
        operator = np.array([[1]])
        for n in range(graph.N):  # build operator to generate state
            if i & (1 << ((graph.N - 1) - n)):
                operator = tensor(operator, Z)
            else:
                operator = tensor(operator, I(2))
        mytuple += (np.dot(operator, gstate),)
    U = H(np.hstack(mytuple))
    return np.dot(np.dot(H(U), rho), U)


def vec_reorder(psi, sys):
    psi = np.copy(psi)  # just to be safe
    n = int(log(psi.shape[0], 2))
    old_shape = psi.shape
    psi = psi.reshape((2,) * n)  # this needs to change for non-qubit systems
    perm = list(sys)
    return psi.transpose(perm).reshape(old_shape)


def reorder(rho, sys):
    # sys is a list that contains a permutation of the subsystems
    rho = np.copy(rho)  # just to be safe
    n = int(log(rho.shape[0], 2))
    old_shape = rho.shape
    rho = rho.reshape((2, 2) * n)  # this needs to change for non-qubit systems
    perm = list(sys) + [x + n for x in sys]
    return rho.transpose(perm).reshape(old_shape)


def ptranspose(rho, sys):
    # sys is a list of the subsystems that should be transposed
    rho = np.copy(rho)  # just to be safe
    n = int(log(rho.shape[0], 2))
    old_shape = rho.shape
    rho = rho.reshape((2, 2) * n)
    perm = np.arange(2 * n)
    for k in sys:
        perm[k] = k + n
        perm[k + n] = k
    return rho.transpose(perm).reshape(old_shape)


def ptrace(rho, sys):
    # sys is a list of subsystems that should be traced over
    rho = np.copy(rho)  # just to be safe
    n = int(log(rho.shape[0], 2))
    # old_shape = rho.shape
    rho = rho.reshape((2, 2) * n)
    sys = np.sort(sys)[::-1]  # sort highest to lowest, so we don't need to re-index after each trace
    for i in sys:
        rho = np.trace(rho, axis1=i, axis2=i + n)
        n -= 1
    return rho.reshape(2**n, 2**n)
