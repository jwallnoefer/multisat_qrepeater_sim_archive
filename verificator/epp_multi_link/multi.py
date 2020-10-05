import numpy as np
import sys
sys.path.append('../../')
sys.path.append('../')
from libs.aux_functions import assert_dir
import os
import luet

z_rot = lambda a, b, c, d: np.array([b, a, d, c])


def dp_doub(t, T, a, b, c, d):
    lam = (1 - np.exp(- t / (2 * T))) / 2
    lam = lam + lam - 2 * lam * lam
    return ((1 - lam) * np.array([a, b, c, d]) + lam * z_rot(a, b, c, d)).tolist()

swap = lambda a, b, c, d, e, f, g, h: np.array([a*e + b*f + c*g + d*h,
       a*f + b*e + c*h + d*g,
       a*g + b*h + c*e + d*f,
       a*h + b*g + c*f + d*e])

def h(x):
	if x == 0:
		return 0
	return -x*np.log2(x)-(1-x)*np.log2(1-x)

def h_derv(x):
	if x == 0:
		return 0
	return -np.log2(x)-np.sign(x)/np.log(2)+np.log2(1-x)+(1-x)*np.sign(1-x)/np.log(2)

def sample(P_link):
	return np.random.geometric(P_link)

def cal_lam_f(l, c, T):
	return (1-np.exp(-l/(c*T))) / 2

def cal_lam_ad(l, c, T, n_2):
	sub = np.exp(-n_2*l/(c*T))
	return (1 - sub) - (1 - 2 * sub + sub**2) / 2

def cal_rho_p_d(lam_ad, lam_f):
	a = lam_ad*lam_f*(1-lam_f)+(1-lam_ad)*(1-lam_f)**2
	c = (1-lam_ad)*lam_f**2+lam_ad*(1-lam_f)*lam_f
	p = a + c
	return [a/p, 0, c/p, 0], p

def create_n_dist(lam_ad, lam_f):
	rho, p = cal_rho_p_d(lam_ad, lam_f)
	cont = np.random.random() <= p
	return cont, rho


class Station():

	def __init__(self, ID, pos):
		self.id = ID
		self.pos = pos


class Link():

	def __init__(self, l_station, r_station, c, T, N = 0, t = 0, rho = [0, 0, 0, 0], got = False, direct = True):
		self.left = l_station
		self.right = r_station
		self.dist = r_station.pos - l_station.pos
		self.stations = (l_station.id, r_station.id)
		self.got = got
		self.direct = direct
		self.N = N
		self.rho = rho
		self.c = c
		self.T = T
		self.t = t

	def create_init(self, P_link):
		if not self.got and self.direct:
			n_1 = sample(P_link)
			n_2 = sample(P_link)
			lam_ad = cal_lam_ad(self.dist, self.c, self.T, n_2)
			lam_f = cal_lam_f(self.dist, self.c, self.T)
			self.got, self.rho = create_n_dist(lam_ad, lam_f)
			self.N += n_1 + n_2
			self.t += 2 * (n_1 + n_2) * self.dist / self.c + self.dist/self.c
			self.rho = dp_doub(self.dist/self.c,self.T, *self.rho)

	def __add__(self, other):
		if not self.stations[1] == other.stations[0]:
			if self.stations[0] == other.stations[1]:
				return other.__add__(self)
			return None
		if not (self.got and other.got):
			return None
		if self.t >= other.t:
			self.rho = dp_doub(self.t-other.t, other.T, *self.rho)
		else:
			other.rho = dp_doub(other.t-self.t, self.T, *other.rho)
		rho = swap(*(self.rho+other.rho))
		t = max(self.t, other.t)
		N = max(self.N, other.N)
		return Link(self.right, other.left, self.c, self.T, N = N, t = t, rho = rho, got = True, direct = False)


class Checker():


	def __init__(self, l, c, T, P_link, num_of_links):
		self.P_link = P_link
		self.l = l
		self.c = c
		self.T = T
		self.num_of_links = num_of_links
		self.N_max = 0
		self.fx = 0
		self.fz = 0
		self.links = []
		self.stations = []
		for i in range(num_of_links + 1):
			self.stations.append(Station(i, i * l / num_of_links))
		for i in range(num_of_links):
			self.links.append(Link(self.stations[i], self.stations[i+1], c, T))

	def reset(self):
		self.links = []
		self.stations = []
		for i in range(self.num_of_links + 1):
			self.stations.append(Station(i, i * self.l / self.num_of_links))
		for i in range(self.num_of_links):
			self.links.append(Link(self.stations[i], self.stations[i+1], self.c, self.T))

	def connect_subs(self, sub):
		t_sorted = sorted(sub, key = lambda l: l.t)
		while len(t_sorted) != 1:
			pivot = t_sorted.pop(0)
			l,r = pivot.stations
			rest = []
			for j in t_sorted:
				if l in j.stations or r in j.stations:
					new_link = pivot + j
					rest.append(new_link)
					break
				else:
					rest.append(j)
			t_sorted = sorted(rest, key = lambda l: l.t)
		return t_sorted[0]




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
				new_links_list.append(self.connect_subs(sub))
			elif len(sub) == 1:
				new_links_list.append(sub[0])
		self.links = new_links_list

	def check_for_create(self):
		new = []
		for l in self.links:
			if not l.got:
				l.create_init(self.P_link)
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
	av = np.sum(arr)/n
	return av, np.sqrt(np.sum(np.abs(arr-av)**2)/n)

P_link = 0.3
T = 10*10**-3
c = 2 * 10**8
l_arr = [1000*i for i in range(5,300,5)]
L_att = 22 * 10**3
n = 4
res = []
for l in l_arr:
	track_list = []
	key_rate_luet = luet.lower_bound(l/2)
	ch = Checker(l,c,T,P_link*np.exp(-l/(n*L_att)),n)
	while len(track_list) < 10000:
		write_in = ch.check()
		if write_in:
			track_list.append([ch.N_max, ch.fx, ch.fz])
	track_list = np.array(track_list).T
	N, dN = av_std(track_list[0])
	fx, dfx = av_std(track_list[1])
	fz, dfz = av_std(track_list[2])
	fraction = (1-h(fx)-h(fz))
	key_rate = fraction / N
	dkr = np.sqrt(((dN*fraction)/N**2)**2 + (h_derv(fx)*dfx/N)**2 + (h_derv(fz)*dfz/N)**2)
	res.append([l, key_rate, dkr, key_rate_luet])
result_path = os.path.join("../../results", "verificator")
assert_dir(result_path)
np.savetxt(os.path.join(result_path, "multi_link_epp4.txt"), np.array(res))
