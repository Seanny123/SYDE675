import numpy as np
import scipy.linalg
import scipy.io
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import ipdb

def majority_vote(a, b, c):
    res = np.zeros(a.shape)
    res[np.where( np.sum((a, b, c)) >= 2 )] = 1
    return res

a = np.zeros(3, dtype=np.bool)
b = np.zeros(3, dtype=np.bool)
c = np.zeros(3, dtype=np.bool)
a[0] = True
b[0] = True
c[-1] = True
print(majority_vote(a, b, c))