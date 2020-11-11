import numpy as np
import matplotlib.pyplot as plt
# equation numbers(?)from title={Overcoming lossy channel bounds using a single quantum repeater node},
# author={Luong, David and Jiang, Liang and Kim, Jungsang and L{\"u}tkenhaus, Norbert},


def alpha(n, p_d):  # function for the detector-efficiency (7)
    return n * (1 - p_d) / (1 - (1 - n) * (1 - p_d)**2)


def h(x):
    if x < 0:
        x = 0
    assert x>=0 and x<= 1, str(x) + " invalid x."
    if x == 0 or x == 1:
        return 0
    return -x*np.log2(x)-(1-x)*np.log2(1-x)


def lower_bound(l):
    l_a = l_b = l
	# getting all the parameters, names like in the section results
    t_p = 0	# preparation time
    t_2 = 1	# dephasing time
    c = 2 * 10**8	# speed of light in fiber
    l_att = 22 * 10**3	# attenuation length
    e_mA = 0.05# misalignment error Alice
    e_mB = 0.05	# misalignment error Bob
    p_d = 0	# dark count probablility per detector
    p_bsm = 1 # Bell state measurement success probability
    lambda_bsm = 1	# Bell state measurement ideality
    f = 1					# error correction inefficiency
    n_tot = 0.3  # (12)
    n_a = n_tot * np.exp(-l_a / l_att)  # (11) wodc = without dark counts
    n_b = n_tot * np.exp(-l_b / l_att)  # (11)
    Y = p_bsm * (1 / n_a + 1 / n_b - 1 / (n_a + n_b - n_a * n_b))**(- 1)  # (14) Y = yield
    tau_b = t_p + 2 * l_b / c  # (19) lower bound for the time elapsed for one round on Bobs side of the QM (preparation + sending + classical signal)
    exp_taub_t2 = np.exp(- tau_b / t_2)
    E_ta = n_a * n_b * np.exp(- 2 * l_a / (c * t_2)) / (n_a + n_b - n_a * n_b)
    E_ta *= (1 / (1 - exp_taub_t2 * (1 - n_a)) + 1 / (1 - exp_taub_t2 * (1 - n_b)) - 1)  # (24)
    eps_dph = 1 / 2 - 1 / 2 * np.exp(- 2 * l_b / (c * t_2)) * E_ta  # can be shown, based on (18)
    eps_m = e_mA * (1 - e_mB) + e_mB * (1 - e_mA)  # (17)
    eff_prod = lambda_bsm * alpha(n_a, p_d) * alpha(n_b, p_d)
    e_x = eff_prod * (eps_m * (1 - eps_dph) + (1 - eps_m) * eps_dph) + 1 / 2 * (1 - eff_prod)  # (15)
    e_z = eff_prod * eps_m + 1 / 2 * (1 - eff_prod)  # (16)
    return Y * (1 - h(e_x) - f * h(e_z))  # (10)



