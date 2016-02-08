import numpy as np
from numpy.linalg import norm, inv
import scipy.linalg
import scipy.io
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import ipdb

def trans_cov(dat, cov, mean=np.array([[0],[0]])):
    evals, evecs = scipy.linalg.eigh(cov)
    c = np.dot(evecs, np.diag(np.sqrt(evals)))
    res = np.dot(c,dat)
    if not np.allclose(np.round(np.cov(res)), cov):
        print(np.cov(res))
    res = res + mean
    assert np.allclose(np.round(np.mean(res, axis=1)), mean.T[0])
    
    return res

def map_class(c1, c2):
    e1 = np.cov(c1)
    e2 = np.cov(c2)
    u1 = np.mean(c1, axis=1)
    u2 = np.mean(c2, axis=1)
    
    def f(x):
        #ipdb.set_trace()
        return np.log(np.sqrt(norm(e2))/np.sqrt(norm(e1))) \
               -0.5*np.dot(np.dot((x - u1),inv(e1)),(x - u1).T) \
               +0.5*np.dot(np.dot((x - u2),inv(e2)),(x - u2).T) \
                < 0
    return f


dats = [
            [trans_cov(np.random.randn(2, 200), np.eye(2)),
             trans_cov(np.random.randn(2, 200), np.eye(2), np.array([[3],[0]]))],
    
            [trans_cov(np.random.randn(2, 200), np.array([[4,3],[3,4]]), np.array([[-1],[0]])),
             trans_cov(np.random.randn(2, 200), np.array([[4,3],[3,4]]), np.array([[1],[0]]))],
    
            [trans_cov(np.random.randn(2, 200), np.array([[3,1],[1,2]])),
             trans_cov(np.random.randn(2, 200), np.array([[7,-3],[-3,4]]), np.array([[3],[0]]))]
        ]

dat = [
    np.array([[-2, 0], [-2, 1], [-2,-1], [3, 0]]).T,
    np.array([[2, 0], [2, 1], [2,-1], [-3, 0]]).T
]

"""
# red left, blue right
fig = plt.figure()
plt.scatter(dat[0][0], dat[0][1], color="red")
plt.scatter(dat[1][0], dat[1][1], color="blue")
plt.show()
"""

map_func = map_class(dat[0], dat[1])
dat_list = list(np.concatenate((dat[0], dat[1]), axis=1).T)

res = []
for d_l in dat_list:
    res.append(map_func(d_l))


# grid the whole space
all_dat = np.array(dat_list)
a_res = np.array(res, dtype=np.int)
a_res[a_res == 0] = -1

blue = all_dat[a_res == -1]
red = all_dat[a_res == 1]
plt.scatter(blue[:, 0], blue[:, 1], color="red")
plt.scatter(red[:, 0], red[:, 1], color="blue")

min_x = np.min(all_dat[:, 0])
max_x = np.max(all_dat[:, 0])
min_y = np.min(all_dat[:, 1])
max_y = np.max(all_dat[:, 1])
delta = 0.01
x = np.arange(min_x, max_x, delta)
y = np.arange(min_y, max_x, delta)
grid_x, grid_y = np.meshgrid(x, y)
#grid_x, grid_y = np.mgrid[min_x:max_x:500j, min_y:max_y:500j]
grid_res = griddata(all_dat, a_res, (grid_x, grid_y), method='cubic')
plt.contour(grid_x.T, grid_y.T, grid_res.T)
plt.show()

# poke values in the correct places
#ipdb.set_trace()