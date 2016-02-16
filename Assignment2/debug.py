import numpy as np
import scipy.linalg
import scipy.io
from numpy.linalg import norm, inv
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

dats = [
            [trans_cov(np.random.randn(2, 200), np.eye(2)),
             trans_cov(np.random.randn(2, 200), np.eye(2), np.array([[3],[0]]))],
    
            [trans_cov(np.random.randn(2, 200), np.array([[4,3],[3,4]]), np.array([[-1],[0]])),
             trans_cov(np.random.randn(2, 200), np.array([[4,3],[3,4]]), np.array([[1],[0]]))],
    
            [trans_cov(np.random.randn(2, 200), np.array([[3,1],[1,2]])),
             trans_cov(np.random.randn(2, 200), np.array([[7,-3],[-3,4]]), np.array([[3],[0]]))]
        ]

def med(c1, c2):
    """Create an med function"""
    z1 = np.mean(c1, axis=1)
    z2 = np.mean(c2, axis=1)
    def f(x):
        return -np.dot(z1,x) + 0.5*np.dot(z1.T,z1) < -np.dot(z2,x) + 0.5*np.dot(z2.T,z2)
    return f

def ged(c1, c2):
    s1 = inv(np.cov(c1))
    s2 = inv(np.cov(c2))
    u1 = np.mean(c1, axis=1)
    u2 = np.mean(c2, axis=1)
    
    def f(x):
        return np.sqrt(np.dot(np.dot((x - u1),s1),(x - u1).T)) > \
               np.sqrt(np.dot(np.dot((x - u2),s2),(x - u2).T)) 
        
    return f

def knn(c1, c2, k, offset=5):
    # initiaslise with all the points from each class
    zero_shape = (c1.shape[0], c1.shape[1]*2)
    c1_res = np.zeros(zero_shape)
    c1_count = 0
    c2_res = np.zeros(zero_shape)
    c2_count = 0
    c_all = np.concatenate((c1[:, offset:], c2[:, offset:]), axis=1)
    err_count = 0

    # because numpy nditer is psychotic
    for c_ind in xrange(c_all.shape[1]):
        val = c_all[:, c_ind]
        # find the nearest K neighbours
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
            
            if c_ind > c1.shape[1]:
                err_count += 1
        else:
            c2_res[:, c2_count] = val
            c2_count += 1
            
            if c_ind <= c1.shape[1]:
                err_count += 1

    assert c1_count + c2_count == c1.shape[1] + c2.shape[1] - 2*offset
    return (c1_res[:, :c1_count], c2_res[:, :c2_count], err_count)

funcs = [med, ged, (knn, 1), (knn, 3), (knn, 5)]
res = [[], [], [], [], []]

def get_err(funct, dat1, dat2, offset=5):
    # get the function, calculate and return the error
    dat_length = dat1.shape[1] + dat2.shape[1]
    if type(funct) is tuple:
        res = funct[0](dat1, dat2, funct[1], offset)
        return float(res[2]) / float(dat_length)
    else:
        func = funct(dat1[:, :offset], dat2[:, :offset])
        tot_err = 0

        if funct == med:
            res = func(dat1[:, offset:])
            tot_err += np.where(res == False)[0].shape[0]
            res = func(dat2[:, offset:])
            tot_err += np.where(res == True)[0].shape[0]

        elif funct == ged:
            res = []
            for dat in list(dat1.T):
                res.append(func(dat))
            a_res = np.array(res, dtype=np.bool)
            tot_err += np.where(a_res == False)[0].shape[0]
            res = []
            for dat in list(dat2.T):
                res.append(func(dat))
            a_res = np.array(a_res, dtype=np.bool)
            tot_err += np.where(res == True)[0].shape[0]

        return float(tot_err) / float(dat_length)


res = [[], [], [], [], []]

# jack-knife
for f_i, funct in enumerate(funcs):
    for dat in dats:
        dat1 = dat[0]
        dat2 = dat[1]

        for offset in xrange(1, 200):
            if funct == ged:
                ipdb.set_trace()

            res[f_i].append(get_err(funct, dat1, dat2, offset))
            
p_err_final = 1/400 * np.sum(res, axis=1)