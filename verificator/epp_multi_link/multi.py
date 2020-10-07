import numpy as np
import sys
sys.path.append('../../')
sys.path.append('../')
from libs.aux_functions import assert_dir
import os
import luet
import Maps as maps
import multiprocessing as mp


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
    l = ch.l / ch.num_of_links
    rho1 = maps.coupl(ch.em, *ch.rho_prep[:])
    rho2 = maps.coupl(ch.em, *ch.rho_prep[:])
    rho1 = maps.dp_sing(2 * l / ch.c, ch.T, *rho1)
    rho2 = maps.dp_sing(2 * l / ch.c, ch.T, *rho2)
    t_wait = n2 * 2 * l / ch.c
    rho1 = maps.dp_doub(t_wait, ch.T, *rho1)
    rho, p = maps.distil(ch.lam_Dist, ch.p_dc,
                         ch.p_dc, *(rho1 + rho2))
    rho = maps.dp_doub(l / ch.c, ch.T, *rho)
    cont = np.random.random() <= p
    return cont, rho


def create(ch, ran_n):
    l = ch.l / ch.num_of_links
    assert np.log2(len(
        ran_n)) % 1 == 0.0, "The number of random variables have to be a power of 2 if binary"
    rho_basic = maps.coupl(ch.em, *ch.rho_prep[:])
    rho_basic = maps.dp_sing(2 * l / ch.c, ch.T, *rho_basic)
    k = int(np.log2(len(ran_n)))
    rhos = [rho_basic[:] for i in range(2**k)]
    p_total = 1
    for i in range(k):
        new_rhos = []
        new_n = []
        for j in range(0, 2**k, 2):
            t_wait = ran_n[j + 1] * 2 * l / ch.c
            rho1 = maps.dp_doub(t_wait, ch.T, *rhos[j])
            rho, p = maps.distil(ch.lam_Dist, ch.p_dc,
                                 ch.p_dc, *(rho1 + rhos[j + 1]))
            p_total *= p
            new_rhos.append(rho)
            new_n.append(ran_n[j] + ran_n[j + 1])
        rhos = new_rhos
        ran_n = new_n
        k = k - 1
    assert len(rhos) == 1, "Okay, something went wrong here, sorry for that!"
    return np.random.random() <= p_total, rhos[0]



class Station():

    def __init__(self, ID, pos):
        self.id = ID
        self.pos = pos


class Link():

    def __init__(self, l_station, r_station, c, T, t_cut=None, N=0, t=0, rho_init=[1, 0, 0, 0], got=False, direct=True, lam_BSM=1, k = 1):
        self.left = l_station
        self.right = r_station
        self.t_cut = t_cut
        self.dist = r_station.pos - l_station.pos
        self.stations = (l_station.id, r_station.id)
        self.got = got
        self.direct = direct
        self.N = N
        self.rho = rho_init
        self.c = c
        self.T = T
        self.t = t
        self.lam_BSM = lam_BSM
        self.k = k

    def create_init(self, P_link, ch):
        if not self.got and self.direct:
            """n_1 = sample(P_link)
            n_2 = sample(P_link)
            self.got, self.rho = create_n_dist(ch, n_2)
            self.N += n_1 + n_2
            self.t += 2 * (n_1 + n_2) * self.dist / self.c + self.dist / self.c"""
            n_vec = [sample(P_link) for i in range(2**self.k)]
            n_sum = sum(n_vec)
            self.got, self.rho = create(ch, n_vec)
            self.N += n_sum
            self.t += 2 * n_sum * self.dist / self.c + 2**(self.k - 1) * self.dist / self.c * (self.k != 0)

    def __add__(self, other):
        if not self.stations[1] == other.stations[0]:
            if self.stations[0] == other.stations[1]:
                return other.__add__(self)
            return None
        if not (self.got and other.got):
            return None
        if self.t_cut != None and np.abs(self.t - other.t) > self.t_cut:
            if self.t > other.t:
                other.got = False
            else:
                self.got = False
            return 0
        if self.t >= other.t:
            self.rho = maps.dp_doub(self.t - other.t, other.T, *self.rho)
        else:
            other.rho = maps.dp_doub(other.t - self.t, self.T, *other.rho)
        rho = maps.swap(self.lam_BSM, *(self.rho + other.rho))
        t = max(self.t, other.t)
        N = max(self.N, other.N)
        return Link(self.right, other.left, self.c, self.T, N=N, t=t, rho_init=rho, got=True, direct=False, t_cut=self.t_cut, lam_BSM=self.lam_BSM, k = self.k)

    def __eq__(self, other):
        return self.stations == other.stations


class Checker():

    def __init__(self, l, c, T, P_link, num_of_links, t_cut=None, em=0, lam_BSM=1, lam_Dist=1, p_dc=0, rho_prep=[1, 0, 0, 0], k=1):
        self.P_link = P_link
        self.l = l
        self.c = c
        self.T = T
        self.num_of_links = num_of_links
        self.k = k
        self.t_cut = t_cut
        self.em = em
        self.P_link = P_link
        self.lam_BSM = lam_BSM
        self.lam_Dist = lam_Dist
        self.p_dc = p_dc
        self.rho_prep = rho_prep
        self.N_max = 0
        self.fx = 0
        self.fz = 0
        self.links = []
        self.stations = []
        for i in range(num_of_links + 1):
            self.stations.append(Station(i, i * l / num_of_links))
        for i in range(num_of_links):
            self.links.append(Link(self.stations[i], self.stations[
                              i + 1], c, T, t_cut, N=0, t=0, rho_init=rho_prep, got=False, direct=True, lam_BSM=lam_BSM, k= k))

    def reset(self):
        self.links = []
        self.stations = []
        for i in range(self.num_of_links + 1):
            self.stations.append(Station(i, i * self.l / self.num_of_links))
        for i in range(self.num_of_links):
            self.links.append(
                Link(self.stations[i], self.stations[i + 1], self.c, self.T, t_cut = self.t_cut, rho_init=self.rho_prep, lam_BSM=self.lam_BSM, k= self.k))

    def catch_sub(self, t_sorted, rest, pivot, j):
        if not pivot.got:
            problem = pivot
        else:
            problem = j
            t_sorted.remove(j)
            rest.append(pivot)
        t = problem.t
        N = problem.N
        i = problem.stations[0]
        end = problem.stations[1]
        replacer = []
        while i != end:
            replacer.append(Link(self.stations[i], self.stations[
                i + 1], c, T, t_cut, N=N, t=t, rho_init=rho_prep, got=False, direct=True, lam_BSM=lam_BSM, k = self.k))
            i += 1
        links = sorted(t_sorted + rest + replacer, key=lambda l: l.stations[0])
        return links

    def connect_subs(self, sub):
        t_sorted = sorted(sub[:], key=lambda l: l.t)
        while len(t_sorted) != 1:
            pivot = t_sorted.pop(0)
            l, r = pivot.stations
            rest = []
            for j in t_sorted:
                if l in j.stations or r in j.stations:
                    new_link = pivot + j
                    if isinstance(new_link, int):
                        return self.catch_sub(t_sorted, rest, pivot, j)
                    rest.append(new_link)
                    break
                else:
                    rest.append(j)
            t_sorted = sorted(rest, key=lambda l: l.t)
        return t_sorted[0:1]

    def check_for_add(self):
        got_list = [l.got for l in self.links]
        if sum(got_list) < 2:
            return None
        i = 0
        new_links_list = []
        while i < len(got_list):
            sub = []
            if not got_list[i]:
                new_links_list.append(self.links[i])
                i += 1
                continue
            while i < len(got_list) and got_list[i]:
                sub.append(self.links[i])
                i += 1
            if len(sub) > 1:
                new_links = self.connect_subs(sub)
                new_links_list += new_links
            elif len(sub) == 1:
                new_links_list.append(sub[0])
        self.links = new_links_list
        self.links = sorted(self.links, key=lambda l: l.stations[0])

    def check_for_create(self):
        new = []
        for l in self.links:
            if not l.got:
                l.create_init(self.P_link, self)
            new.append(l)
        self.links = new

    def check(self):
        self.check_for_create()
        self.check_for_add()
        if len(self.links) == 1:
            rho = self.links[0].rho
            self.N_max = self.links[0].N
            self.fx = rho[1] + rho[3]
            self.fz = rho[2] + rho[3]
            self.reset()
            return True
        return False


def av_std(arr):
    n = np.shape(arr)[0]
    av = np.sum(arr) / n
    return av, np.sqrt(np.sum(np.abs(arr - av)**2) / n)


result_path = os.path.join("../../results", "verificator")
assert_dir(result_path)


def runner(kwargs):
    assert 'n' in kwargs and 'k' in kwargs, "One meta-parameter missing"
    P_link = 0.3
    T = 1
    c = 2 * 10**8
    em = 0.05
    l_arr = [1000 * i for i in range(10, 510, 10)]
    L_att = 22 * 10**3
    n = kwargs.pop('n', None)
    k = kwargs['k']
    res = []
    for l in l_arr:
        track_list = []
        ch = Checker(l, c, T, P_link * np.exp(-l / (n * L_att)), n, em=em, **kwargs)
        while len(track_list) < 10000:
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
        res.append([l, key_rate, dkr])
        if key_rate <= 0:
            break
    np.savetxt(os.path.join(result_path, "multi_link_epp" + str(n) + "_" + str(k) + ".txt"), np.array(res))
    return 

kwargs_tuple = [(1,1),(1,2),(2,1),(2,2),(3,1),(3,2),(4,1),(4,2),(5,1),(5,2),(6,1),(6,2),(7,1),(7,2),(8,1),(8,2),(5,0),(6,0),(7,0),(8,0)]
#kwargs_tuple = [(4,2)]
kwargs_list = [{'n':n, 'k':k} for n,k in kwargs_tuple]
with mp.Pool(min(len(kwargs_list),mp.cpu_count())) as pool:
    pool.map(runner, kwargs_list)
