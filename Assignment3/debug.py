import numpy as np
import scipy.linalg
import scipy.io
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from functools import partial
import ipdb

np.random.seed(seed=1)

def med(c1, c2):
    """Create an med function"""
    z1 = np.mean(c1, axis=1)
    z2 = np.mean(c2, axis=1)
    def f(x):
        return -np.dot(z1,x) + 0.5*np.dot(z1.T,z1) < -np.dot(z2,x) + 0.5*np.dot(z2.T,z2)
    return f

def choose(dat, size):
    try:
        return dat[:, np.random.choice(dat.shape[1], size=size, replace=False)]
    except:
        ipdb.set_trace()

def get_diff(a, b):
    return np.where(a != b)[0]

def majority_vote(a, b, c):
    res = np.zeros(a.shape)
    res[np.where( np.sum((a, b, c), axis=0) >= 2 )] = 1
    return res

def make_booster(a_dat, b_dat, clsfier):
    # choose a quarter for training and classify
    a_err = np.empty((2,0))
    b_err = np.empty((2,0))
    while b_err.shape[1] <= 1 or a_err.shape[1] <= 1:
        c1 = clsfier(choose(a_dat, size=a_dat.shape[1]/4), choose(b_dat, size=b_dat.shape[1]/4))

        a_err = a_dat[:, np.where(c1(a_dat) == False)[0]]
        b_err = b_dat[:, np.where(c1(b_dat) == True)[0]]

    # choose a half of the erroroneously classed and train
    c2 = clsfier(choose(a_err, size=a_err.shape[1]/2), choose(b_err, size=b_err.shape[1]/2))

    # train on disagreement of c1 and c2
    a_c1 = c1(a_dat)
    a_c2 = c2(a_dat)
    b_c1 = c1(b_dat)
    b_c2 = c2(b_dat)
    
    a_diff = a_dat[:, get_diff(a_c1, a_c2)]
    b_diff = b_dat[:, get_diff(b_c1, b_c2)]
    c3 = clsfier(a_diff, b_diff)
    
    def boost_eval(x):
        return majority_vote(c1(x), c2(x), c3(x))
    
    return boost_eval

def make_q_class(c1, c2, q):
    err = []
    cls = []
    
    for q_i in xrange(q):
        # get the error
        tot_err = 0
        a = choose(c1, 1)
        b = choose(c2, 1)
        cls.append(med(a, b))
        
        tot_err += np.where(cls[-1](c1) == False)[0].shape[0]
        tot_err += np.where(cls[-1](c2) == True)[0].shape[0]
        err.append(tot_err)

    return cls[np.argmin(err)]

def get_med_err(a, b, med_func):
    tot_err = 0
    tot_err += np.where(med_func(a) == False)[0].shape[0]
    tot_err += np.where(med_func(b) == True)[0].shape[0]
    return tot_err

mat = scipy.io.loadmat("assign3.mat")
dat = [mat["a"].T, mat["b"].T]

# explore the range of q, from 5 to 30
q_err = []
for q_val in range(5, 35, 5):
    q_err.append([])
    for _ in range(10):
        q_tmp = make_booster(dat[0], dat[1], partial(make_q_class, q=q_val))
        q_err[-1].append(get_med_err(dat[0], dat[1], q_tmp))
print(q_err)