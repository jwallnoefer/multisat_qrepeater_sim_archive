import numpy as np
import sys
sys.path.append('../../')
sys.path.append('../')
from libs.aux_functions import assert_dir
import os
import luet
import Maps as maps


def h(x):
    if x == 0:
        return 0
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)


def h_derv(x):
    if x == 0:
        return 0
    return -np.log2(x) - np.sign(x) / np.log(2) + np.log2(1 - x) + (1 - x) * np.sign(1 - x) / np.log(2)


def sample(P_link):
    return np.random.geometric(P_link)


def create_n_dist(ch, n2):
    rho1 = maps.coupl(ch.em, *ch.rho_prep[:])
    rho2 = maps.coupl(ch.em, *ch.rho_prep[:])
    rho1 = maps.dp_sing(2 * ch.l / ch.c, ch.T, *rho1)
    rho2 = maps.dp_sing(2 * ch.l / ch.c, ch.T, *rho2)
    t_wait = n2 * 2 * ch.l / ch.c
    rho1 = maps.dp_doub(t_wait, ch.T, *rho1)
    rho, p = maps.distil(ch.lam_Dist, ch.p_dc_outer,
                         ch.p_dc_inner, *(rho1 + rho2))
    rho = maps.dp_doub(ch.l / ch.c, ch.T, *rho)
    cont = np.random.random() <= p
    return cont, rho


class Checker():

    def __init__(self, l, c, T, P_link, t_cut=None, em=0, lam_BSM=1, lam_Dist=1, p_dc_inner=0, p_dc_outer=0, rho_prep=[1, 0, 0, 0]):
        self.l = l
        self.c = c
        self.T = T
        self.t_cut = t_cut
        self.em = em
        self.P_link = P_link
        self.lam_BSM = lam_BSM
        self.lam_Dist = lam_Dist
        self.p_dc_inner = p_dc_inner
        self.p_dc_outer = p_dc_outer
        self.rho_prep = rho_prep
        self.N_L = 0
        self.N_R = 0
        self.t_L = 0
        self.t_R = 0
        self.got_left = False
        self.got_right = False
        self.rho_L = [0, 0, 0, 0]
        self.rho_R = [0, 0, 0, 0]
        self.N_max = 0
        self.fx = 0
        self.fz = 0

    def check(self):
        if not self.got_left:
            n_L_1 = sample(self.P_link)
            n_L_2 = sample(self.P_link)
            if self.t_cut == None or not n_L_2 * 2 * self.l / self.c > self.t_cut:
                self.got_left, self.rho_L = create_n_dist(self, n_L_2)
            self.t_L += self.l / self.c
            self.t_R += 2 * (n_L_1 + n_L_2) * self.l / self.c
            self.N_L += n_L_1 + n_L_2
        if not self.got_right:
            n_R_1 = sample(self.P_link)
            n_R_2 = sample(self.P_link)
            if self.t_cut == None or not n_R_2 * 2 * self.l / self.c > self.t_cut:
                self.got_right, self.rho_R = create_n_dist(self, n_R_2)
            self.t_R += self.l / self.c
            self.t_R += 2 * (n_R_1 + n_R_2) * self.l / self.c
            self.N_R += n_R_1 + n_R_2
        if self.got_right and self.got_left:
            if self.t_cut != None and np.abs(self.t_L - self.t_R) > self.t_cut:
                if self.t_L < self.t_R:
                    self.got_left = False
                    return False
                else:
                    self.got_right = False
                    return False
            self.N_max = max(self.N_L, self.N_R)
            if self.N_L >= self.N_R:
                self.rho_L = maps.dp_doub(2 * (self.N_L - self.N_R)
                                          * self.l / self.c, self.T, *self.rho_L)
            else:
                self.rho_R = maps.dp_doub(2 * (self.N_R - self.N_L)
                                          * self.l / self.c, self.T, *self.rho_R)
            rho = maps.swap(self.lam_BSM, *(self.rho_L + self.rho_R))
            self.fx = rho[1] + rho[3]
            self.fz = rho[2] + rho[3]
            self.N_L = 0
            self.N_R = 0
            self.t_L = 0
            self.t_R = 0
            self.got_right = False
            self.got_left = False
            return True
        return False


def av_std(arr):
    n = np.shape(arr)[0]
    av = np.sum(arr) / n
    return av, np.sqrt(np.sum(np.abs(arr - av)**2) / n)

P_link = 0.3
T = 1
c = 2 * 10**8
em = 0.05
l_arr = [1000 * i for i in range(5, 300, 5)]
L_att = 22 * 10**3
res = []
fx_list = []
fz_list = []
for l in l_arr:
    print(l)
    l = l / 2
    track_list = []
    key_rate_luet = luet.lower_bound(l)
    ch = Checker(l, c, T, P_link * np.exp(-l / L_att), em=em)
    while len(track_list) < 20000:
        write_in = ch.check()
        if write_in:
            track_list.append([ch.N_max, ch.fx, ch.fz])
    track_list = np.array(track_list).T
    N, dN = av_std(track_list[0])
    fx, dfx = av_std(track_list[1])
    fz, dfz = av_std(track_list[2])
    fraction = (1 - h(fx) - h(fz))
    key_rate = fraction / N
    dkr = np.sqrt(((dN * fraction) / N**2)**2 + (h_derv(fx)
                                                 * dfx / N)**2 + (h_derv(fz) * dfz / N)**2)
    res.append([2 * l, key_rate, dkr, key_rate_luet])
    fx_list.append(fx)
    fz_list.append(fz)
result_path = os.path.join("../../results", "verificator")
assert_dir(result_path)
np.savetxt(os.path.join(result_path, "epp_two_link.txt"), np.array(res))
np.savetxt(os.path.join(result_path, "fx_list.txt"), np.array(fx_list))
np.savetxt(os.path.join(result_path, "fz_list.txt"), np.array(fz_list))
