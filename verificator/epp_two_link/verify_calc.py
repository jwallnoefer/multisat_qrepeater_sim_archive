import os
import numpy as np
import libs.matrix as mat
import sys
sys.path.append('../../')
import sympy as sp
from sympy.physics.quantum import TensorProduct
from sympy.physics.quantum.operator import Operator
from sympy.physics.quantum.dagger import Dagger
from sympy.parsing.sympy_parser import parse_expr


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
    state = rho  # renormalize
    return p_suc, state


sqrt_plus_ix = 1 / np.sqrt(2) * (mat.I(2) + 1j * mat.X)
sqrt_minus_ix = mat.H(sqrt_plus_ix)
bilateral_cnot = np.dot(mat.CNOT(0, 2, N=4), mat.CNOT(1, 3, N=4))
dejmps_operator = np.dot(bilateral_cnot, mat.tensor(sqrt_minus_ix, sqrt_plus_ix, sqrt_minus_ix, sqrt_plus_ix))
dejmps_operator_dagger = mat.H(dejmps_operator)
dejmps_proj_ket_z0z0 = mat.tensor(mat.I(4), mat.z0, mat.z0)
dejmps_proj_ket_z1z1 = mat.tensor(mat.I(4), mat.z1, mat.z1)
dejmps_proj_bra_z0z0 = mat.H(dejmps_proj_ket_z0z0)
dejmps_proj_bra_z1z1 = mat.H(dejmps_proj_ket_z1z1)

lam_f, lam_ad = sp.symbols('L_f L_ad')

z_rot = lambda a, b, c, d: (b, a, d, c)

def sp_z_rot(rho):
    args = np.diag(rho).tolist()
    return sp.diag(*z_rot(*args))
#init 1
rho_1 = sp.diag(1,0,0,0)
#deph while sending
rho_1 = (1-lam_f)*rho_1+lam_f*sp_z_rot(rho_1)
#deph while waiting for second pair:
rho_1 = (1-lam_ad)*rho_1+lam_ad*sp_z_rot(rho_1)

#init 2
rho_2 = sp.diag(1,0,0,0)
#deph while sending
rho_2 = (1-lam_f)*rho_2+lam_f*sp_z_rot(rho_2)

sq = np.sqrt(2)
T = np.array([[1,0,0,1],[1,0,0,-1],[0,1,1,0],[0,1,-1,0]])/sq
T_sp = sp.Matrix(T)
TT_sp = TensorProduct(T_sp, T_sp)
four_bit_state = Dagger(TT_sp)*TensorProduct(rho_1, rho_2)*TT_sp
p, rho_pdist = dejmps_protocol(four_bit_state)
print(np.diag(T_sp*rho_pdist*Dagger(T_sp)))

