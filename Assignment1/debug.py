import numpy as np
from numpy.linalg import norm, inv
import scipy.linalg
import scipy.io
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
        print(val)
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

res_1, res_2 = knn(dat[0], dat[1], 3)
fig = plt.figure()
plt.scatter(res_1[0], res_1[1], color="red")
plt.scatter(res_2[0], res_2[1], color="blue")
plt.show()