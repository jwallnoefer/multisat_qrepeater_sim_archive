import numpy as np
import sys
sys.path.append('../../')
from libs.aux_functions import assert_dir
import os

z_rot = lambda a, b, c, d: np.array([b, a, d, c])


def dp_doub(t, T, a, b, c, d):
    lam = (1 - np.exp(- t / (2 * T))) / 2
    lam = lam + lam - 2 * lam * lam
    return ((1 - lam) * np.array([a, b, c, d]) + lam * z_rot(a, b, c, d)).tolist()

def swap(a, b, c, d, e, f, g, h):
    state = np.array([a*e/4 + b*f/4 + c*g/4 + d*h/4,
                      a*f/4 + b*e/4 + c*h/4 + d*g/4,
                      a*g/4 + b*h/4 + c*e/4 + d*f/4,
                      a*h/4 + b*g/4 + c*f/4 + d*e/4])
    return state / np.sum(state)

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

class Checker():


	def __init__(self, l, c, T, P_link):
		self.l = l
		self.c = c
		self.T = T
		self.lam_f = cal_lam_f(l, c, T)
		self.P_link = P_link
		self.N_L = 0
		self.N_R = 0
		self.got_left = False
		self.got_right = False
		self.rho_L = [0,0,0,0]
		self.rho_R = [0,0,0,0]
		self.N_max = 0
		self.fx = 0
		self.fz = 0

	def check(self):
		if not self.got_left:
			n_L_1 = sample(self.P_link)
			n_L_2 = sample(self.P_link)
			lam_ad_L = cal_lam_ad(self.l, self.c, self.T, n_L_2)
			self.got_left, self.rho_L = create_n_dist(lam_ad_L, self.lam_f)
			self.N_L += n_L_1 + n_L_2
		if not self.got_right:
			n_R_1 = sample(self.P_link)
			n_R_2 = sample(self.P_link)
			lam_ad_R = cal_lam_ad(self.l, self.c, self.T, n_R_2)
			self.got_right, self.rho_R = create_n_dist(lam_ad_R, self.lam_f)
			self.N_R += n_R_1 + n_R_2
		if self.got_right and self.got_left:
			self.N_max = max(self.N_L, self.N_R)
			if self.N_L>=self.N_R:
				self.rho_L = dp_doub(2*(self.N_L - self.N_R + 1/2)*self.l/self.c,self.T,*self.rho_L)
				self.rho_R = dp_doub(self.l/self.c,self.T,*self.rho_R) # time to communicate the
			else:
				self.rho_R = dp_doub(2*(self.N_R - self.N_L + 1/2)*self.l/self.c,self.T,*self.rho_R)
				self.rho_L = dp_doub(self.l/self.c,self.T,*self.rho_L)
			rho = swap(*(self.rho_L+self.rho_R))
			self.fx = rho[1] + rho[3]
			self.fz = rho[2] + rho[3]
			self.N_L = 0
			self.N_R = 0
			self.got_right = False
			self.got_left = False
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
res = []
fx_list = []
fz_list = []
for l in l_arr:
	print(l)
	l = l/2
	track_list = []
	ch = Checker(l,c,T,P_link*np.exp(-l/L_att))
	while len(track_list) < 20000:
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
	res.append([2*l, key_rate, dkr])
	fx_list.append(fx)
	fz_list.append(fz)
result_path = os.path.join("../../results", "verificator")
assert_dir(result_path)
np.savetxt(os.path.join(result_path, "epp_two_link.txt"), np.array(res))
np.savetxt(os.path.join(result_path, "fx_list.txt"), np.array(fx_list))
np.savetxt(os.path.join(result_path, "fz_list.txt"), np.array(fz_list))
