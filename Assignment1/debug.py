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

def knn(c1, c2, k):
    # initiaslise with K points from each class
    zero_shape = (c1.shape[0], c1.shape[1]*2)
    c1_res = np.zeros(zero_shape)
    c1_count = 0
    c2_res = np.zeros(zero_shape)
    c2_count = 0
    c_all = np.concatenate((c1, c2), axis=1)

    # because numpy nditer is psychotic
    for c_ind in xrange(c_all.shape[1]):
        val = c_all[:, c_ind]
        #find the nearest K neighbours
        if np.allclose(val, np.array([-2, 0])):
            ipdb.set_trace()
        ind = np.argpartition(norm(c_all.T - val, axis=1), k+1)[:k+1][1:k+1]
        
        # class the point where the majority of the neighbours are
        sort_res = 0
        for ix in ind:
            if ix < c1.shape[1]:
                sort_res += 1
            else:
                sort_res -= 1

        if sort_res > 0:
            c1_res[:, c1_count] = val
            c1_count += 1
        else:
            c2_res[:, c2_count] = val
            c2_count += 1

    assert c1_count + c2_count == c1.shape[1] + c1.shape[1]
    return (c1_res[:, :c1_count], c2_res[:, :c2_count])


dats = [
            [trans_cov(np.random.randn(2, 200), np.eye(2)),
             trans_cov(np.random.randn(2, 200), np.eye(2), np.array([[3],[0]]))],
    
            [trans_cov(np.random.randn(2, 200), np.array([[4,3],[3,4]]), np.array([[-1],[0]])),
             trans_cov(np.random.randn(2, 200), np.array([[4,3],[3,4]]), np.array([[1],[0]]))],
    
            [trans_cov(np.random.randn(2, 200), np.array([[3,1],[1,2]])),
             trans_cov(np.random.randn(2, 200), np.array([[7,-3],[-3,4]]), np.array([[3],[0]]))]
        ]

"""
dat = [
    np.array([[-2, 0], [-2, 1], [-2,-1], [3, 0]]).T,
    np.array([[2, 0], [2, 1], [2,-1], [-3, 0]]).T
]
"""
dat = dats[2]
all_dat = np.concatenate((dat[0], dat[1]), axis=1).T
min_x = np.min(all_dat[:, 0])
max_x = np.max(all_dat[:, 0])
min_y = np.min(all_dat[:, 1])
max_y = np.max(all_dat[:, 1])
sample_points = 1000
x = np.linspace(min_x, max_x, sample_points)
y = np.linspace(min_y, max_y, sample_points)
grid_x, grid_y = np.meshgrid(x, y)

res1, res2 = knn(dat[0], dat[1], 1)
a_res = np.concatenate((np.ones(res1.shape[1]), np.ones(res2.shape[1])*-1))
grid_res = griddata(all_dat, a_res, (grid_x, grid_y), method='nearest')
print(grid_res.shape)
plt.contour(grid_x.T, grid_y.T, grid_res.T, levels=[0], colors='blue')