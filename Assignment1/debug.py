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
    c_orig = np.concatenate((c1, c2), axis=1)
    c_all = np.concatenate((c1, c2), axis=1)
    c1_del_count = 0

    # because numpy nditer is psychotic
    for c_ind in xrange(c_orig.shape[1]):
        val = c_orig[:, c_ind]
        #find the nearest K neighbours
        print(val)
        #ipdb.set_trace()
        ind = np.argpartition(norm(c_all.T - val, axis=1), k+1)[:k+1][1:k+1]
        
        # class the point where the majority of the neighbours are
        sort_res = 0
        for ix in ind:
            if ix < c1.shape[0]:
                sort_res += 1
            else:
                sort_res -= 1

        # because c2_del_count has no effect
        actual_ind = c_ind - c1_del_count
        if sort_res > 0 and actual_ind > c1.shape[1]:
            print("Swapping: %s" %val)
            c2 = np.delete(c2, (actual_ind - c1.shape[1]), axis=1)
            ipdb
            c1 = np.append(c1, val).reshape((2,-1))
        elif sort_res < 0 and actual_ind < c1.shape[1]:
            print("Swapping: %s" %val)
            c1 = np.delete(c1, actual_ind, axis=1)
            c2 = np.append(c2, val).reshape((2,-1))

        c_all = np.concatenate((c1, c2), axis=1)

    return (c1, c2)


dats = [
            [trans_cov(np.random.randn(2, 200), np.eye(2)),
             trans_cov(np.random.randn(2, 200), np.eye(2), np.array([[3],[0]]))],
    
            [trans_cov(np.random.randn(2, 200), np.array([[4,3],[3,4]]), np.array([[-1],[0]])),
             trans_cov(np.random.randn(2, 200), np.array([[4,3],[3,4]]), np.array([[1],[0]]))],
    
            [trans_cov(np.random.randn(2, 200), np.array([[3,1],[1,2]])),
             trans_cov(np.random.randn(2, 200), np.array([[7,-3],[-3,4]]), np.array([[3],[0]]))]
        ]

dat = [
    np.array([[-1, 0], [-1, 1], [-1,-1], [2, 0]]).T,
    np.array([[1, 0], [1, 1], [1,-1], [-2, 0]]).T
]

res_1, res_2 = knn(dat[0], dat[1], 1)
fig = plt.figure()
plt.scatter(res_1[0], res_1[1], color="red")
plt.scatter(res_2[0], res_2[1], color="blue")
plt.show()
ipdb.set_trace()