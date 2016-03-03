import numpy as np
import scipy.linalg
import scipy.io
from numpy.linalg import norm, inv
from itertools import chain
import ipdb

dats = []
mat = scipy.io.loadmat("assign2.mat")
dats.append([mat["a1"], mat["b1"]])
dats.append([mat["a2"], mat["b2"]])
dats.append([mat["a3"], mat["b3"]])
dats.append([mat["a4"], mat["b4"]])

def med(c1, c2):
    """Create an med function"""
    z1 = np.mean(c1, axis=1)
    assert z1.shape == (2,)
    z2 = np.mean(c2, axis=1)
    assert z2.shape == (2,)

    def f(x):
        return -np.dot(z1,x) + 0.5*np.dot(z1.T,z1) < -np.dot(z2,x) + 0.5*np.dot(z2.T,z2)

    return f

def ged(c1, c2):
    s1 = inv(np.cov(c1))
    assert s1.shape == (2,2)
    s2 = inv(np.cov(c2))
    assert s2.shape == (2,2)
    u1 = np.mean(c1, axis=1)
    assert u1.shape == (2,)
    u2 = np.mean(c2, axis=1)
    assert u2.shape == (2,)

    def f(x):
        return np.sqrt(np.dot(np.dot((x - u1),s1),(x - u1).T)) > \
               np.sqrt(np.dot(np.dot((x - u2),s2),(x - u2).T)) 

    return f

# take a deep breath and unit test KNN with dummy data like we had before

def knn(c1, c2, k, offset=5):
    # initiaslise with all the points from each class
    c1_len = c1.shape[1]
    zero_shape = (c1.shape[0], c1_len*2)
    c1_res = np.zeros(zero_shape)
    c1_count = 0
    c2_res = np.zeros(zero_shape)
    c2_count = 0
    c_train = np.concatenate((c1[:, :offset], c2[:, :offset]), axis=1)
    c_all = np.concatenate((c1, c2), axis=1)
    err_count = 0

    # iterate through all the test points using indexing
    # because numpy nditer is psychotic
    print("Offset:%s c1_len:%s c1_len_offset:%s" %(offset, c1_len, c1_len+offset))
    for c_ind in chain(xrange(offset, c1_len), xrange(c1_len+offset, c1_len*2)):
        val = c_all[:, c_ind]
        # find the nearest K neighbours
        ind = np.argpartition(norm(c_train.T - val, axis=1), k+1)[:k+1][1:k+1]

        # class the point where the majority of the neighbours are
        sort_res = 0
        for ix in ind:
            if ix < offset:
                sort_res += 1
            else:
                sort_res -= 1

        print("c_ind: %s" %c_ind)
        print("val: %s" %val)
        print("closest: %s" %c_train[:, ind[0]])
        print("sort_res: %s" %sort_res)
        #ipdb.set_trace()

        if sort_res > 0:
            c1_res[:, c1_count] = val
            c1_count += 1

            if c_ind > c1_len:
                err_count += 1
        else:
            c2_res[:, c2_count] = val
            c2_count += 1

            if c_ind <= c1_len:
                err_count += 1

    assert c1_count + c2_count == 2*(c1_len-offset)
    return (c1_res[:, :c1_count], c2_res[:, :c2_count], err_count)

def get_err(funct, dat1, dat2, train=5):

    # get the function, calculate and return the error
    tot_res = 2.0*float(dat1.shape[1] - train)
    assert tot_res >= 2

    if type(funct) is tuple:
        res = funct[0](dat1, dat2, funct[1], train)
        assert float(res[2]) <= tot_res
        return float(res[2]) / tot_res
    else:
        func = funct(dat1[:, :train], dat2[:, :train])
        tot_err = 0

        if funct == med:
            res = func(dat1[:, train:])
            tot_err += np.where(res == False)[0].shape[0]
            res = func(dat2[:, train:])
            tot_err += np.where(res == True)[0].shape[0]

        elif funct == ged:

            def ged_err(ged_func, ged_dat, expected):
                res = []
                for dat in list(ged_dat):
                    res.append(ged_func(dat))
                res = np.array(res, dtype=np.bool)
                return np.where(res == expected)[0].shape[0]

            tot_err += ged_err(func, dat1[:, train:].T, True)
            tot_err += ged_err(func, dat2[:, train:].T, False)

        assert tot_err <= tot_res
        return float(tot_err) / tot_res

dat = [
    np.array([[-2, 0], [-2, 1], [-2,-1], [3, 0]]).T,
    np.array([[2, 0], [2, 1], [2,-1], [-3, 0]]).T
]
res = []

dat1 = dat[0]
dat2 = dat[1]

for _ in xrange(dat1.shape[1]):
    res.append(get_err((knn, 1), dat1, dat2, dat1.shape[1]-1))
    dat1 = np.roll(dat1, 1, axis=1)
    dat2 = np.roll(dat2, 1, axis=1)

ipdb.set_trace()