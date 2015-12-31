"""
This file is part of the DiffractionMicroscopy project.
Copyright 2015 David W. Hogg (NYU).

## Bugs:
- I ought to cache (but carefully) previously called marginalized likelihoods b/c Powell sux.
-- The cache will get destroyed when the data are changed.
-- The cache will get destroyed when the orientation sampling is changed.

## Comments:
- I hard-coded some 2x2 linear algebra for speed.
"""
import numpy as np
import scipy.optimize as op
from scipy.misc import logsumexp
import os, errno, glob
import pickle as cp

if True:
    from multiprocessing import Pool
    p = Pool(8)
    pmap = p.map
else:
    pmap = map

Truth = 1. / np.array([47., 13., 11.]) # axis-aligned inverse variances

class GaussianMolecule():

    def __init__(self):
        self.data = None
        self.ivar = None
        self.ivarts = None
        self.lndets = None
        self.Ps = None
        return None

    def set_data(self, data):
        N, K, two = data.shape
        assert two == 2
        self.N = N
        self.K = K
        self.data = data
        return None

    def get_data(self):
        assert self.data is not None
        return self.data

    def set_ivar(self, ivar):
        assert ivar.shape == (3, )
        assert np.all(ivar > 0.)
        self.ivar = ivar
        self.ivarts = None # reset
        self.lndets = None
        return None

    def get_ivar(self):
        return self.ivar

    def set_ivar_from_vector(self, vector):
        self.set_ivar(np.exp(vector))
        return None

    def get_vector(self, vector):
        return np.log(self.ivar)

    def _normalize_vectors(self, vecs):
        return vecs / np.sqrt(np.sum(vecs * vecs, axis=1))[:,None]

    def make_projection_matrix_sampling(self, T):
        self.T = T
        xhat = self._normalize_vectors(np.random.normal(size=(self.T, 3)))
        assert np.allclose(np.sum(xhat * xhat, axis = 1), 1.)
        yhat = np.random.normal(size=(self.T, 3))
        yhat = self._normalize_vectors(yhat - xhat * np.sum(xhat * yhat, axis=1)[:,None])
        assert np.allclose(np.sum(xhat * yhat, axis = 1), 0.)
        assert np.allclose(np.sum(yhat * yhat, axis = 1), 1.)
        self.Ps = np.hstack((xhat, yhat)).reshape((self.T,2,3))
        self.ivarts = None # reset
        self.lndets = None
        return None

    def get_Ps(self):
        if self.Ps is None:
            self.make_projection_matrix_sampling(2048) # magic
        return self.Ps

    def get_ivarts(self):
        """
        Hard-coded for speed.
        """
        if self.ivarts is None:
            Ps = self.get_Ps()
            ivar = self.get_ivar()
            ivarts = np.zeros((self.T, 2, 2))
            ivarts[:, 0, 0] = np.sum(Ps[:,0,:] * ivar[None,None,:] * Ps[:,0,:], axis=2)
            foo             = np.sum(Ps[:,0,:] * ivar[None,None,:] * Ps[:,1,:], axis=2)
            ivarts[:, 0, 1] = foo
            ivarts[:, 1, 0] = foo
            ivarts[:, 1, 1] = np.sum(Ps[:,1,:] * ivar[None,None,:] * Ps[:,1,:], axis=2)
            self.ivarts = ivarts
            self.lndets = None
        return self.ivarts

    def get_lndets(self):
        """
        Hard-coded for speed.
        NOT SAFE FOR UNDERFLOW.
        """
        if self.lndets is None:
            ivarts = self.get_ivarts()
            self.lndets = np.log(ivarts[:,0,0] * ivarts[:,1,1] - ivarts[:,0,1] * ivarts[:,1,0])
        return self.lndets

    def get_datum(self, n):
        return self.get_data()[n]

    def get_chisquareds(self, n):
        dd = self.get_datum(n)
        if n == 0:
            foo = np.sum(np.sum(np.tensordot(dd, self.get_ivarts(), axes=(1,1)) * dd[:,None,:],
                                axis=2), axis=0)
        return np.sum(np.sum(np.tensordot(dd, self.get_ivarts(), axes=(1,1)) * dd[:,None,:],
                             axis=2), axis=0) # should be length T

    def _marginalized_ln_likelihood_one(self, n):
        """
        Compute the sampling approximation to the marginalized likelihood.
        """
        return logsumexp(-0.5 * self.get_chisquareds(n) + 0.5 * self.K * self.get_lndets())
        # K for multiplicity; plus because INVERSE variances

    def marginalized_ln_likelihood(self):
        """
        TOTALLY CONFUSED ABOUT WHAT map() DOES.
        """
        lnlikes = [self._marginalized_ln_likelihood_one(n) for n in range(self.N)]
        return np.sum(lnlikes)

    def __call__(self, vector, verbose=True):
        self.set_ivar_from_vector(vector)
        result = -2. * self.marginalized_ln_likelihood()
        if verbose: print("model({}) = {}".format(1./self.get_ivar(), result))
        return result

def make_fake_data(N, K):
    """
    Every image contains exactly K photons.
    This is unrealistic, but suck it up.

    # Notes:
    - This punks the GaussianMolecule() class to make fake data.
    """
    foo = GaussianMolecule()
    foo.set_ivar(Truth)
    foo.make_projection_matrix_sampling(N)
    ivarns = foo.get_ivarts()
    zero = np.zeros(2)
    samples = np.zeros((N, K, 2))
    for n in range(N):
        Vn = np.linalg.inv(ivarns[n])
        samples[n] = np.random.multivariate_normal(zero, Vn, size=K)
    return samples

def pickle_to_file(fn, stuff):
    fd = open(fn, "wb")
    cp.dump(stuff, fd)
    print("writing", fn)
    fd.close()

def read_pickle_file(fn):
    fd = open(fn, "rb")
    stuff = cp.load(fd)
    fd.close()
    return stuff

def mkdir_p(path):
    """
    This is copied from http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
    """
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def make_and_fit_one_model(model, log2NK, log2K, iteration):
    log2N = log2NK - log2K
    dirname = "./{:02d}".format(iteration)
    mkdir_p(dirname)
    prefix = "{:02d}_{:02d}_{:02d}".format(log2N,log2K,iteration)
    picklefn = dirname + "/model_" + prefix + ".pkl"
    if log2N < 2:
        print(prefix, "is an absurd case; skipping")
        return None # don't do absurd cases
    if len(glob.glob(picklefn)) > 0:
        print(prefix, "already exists; skipping")
        return None # noclobber
    print("starting run", prefix)
    
    # lay down a "lock file"
    pickle_to_file(picklefn, 0.)

    # make fake data
    np.random.seed((iteration + 23) * log2NK)
    data = make_fake_data(N=2**log2N, K=2**log2K)
    model.set_data(data)

    # initialize empirically
    empvar = np.mean(data * data)
    x0 = np.log(1. / np.array([1.1 * empvar, empvar, 0.9 * empvar]))
    x0 = np.sort(x0)
    
    # optimize
    direc = np.array([[1., 1., 1.], [1., 0., -1.], [-1., 2., -1.]]) / 10.
    def bar(x): print(prefix, x, np.exp(-x))
    x1 = op.fmin_powell(model, x0, callback=bar, direc=direc, xtol=1.e-3, ftol=1.e-5)
    x1 = np.sort(x1)
    x2 = op.fmin_powell(model, x1, callback=bar, direc=direc, xtol=1.e-5, ftol=1.e-5)
    x2 = np.sort(x2)

    # check size of P sampling
    sixf = np.zeros(6)
    sixf[0] = model(x2[[0,1,2]])
    sixf[1] = model(x2[[1,2,0]])
    sixf[2] = model(x2[[2,0,1]])
    sixf[3] = model(x2[[2,1,0]])
    sixf[4] = model(x2[[1,0,2]])
    sixf[5] = model(x2[[0,2,1]])

    # save
    model.set_ivar_from_vector(x2) # restore final answer
    pickle_to_file(picklefn, (model, x0, x1, x2, sixf))
    print(prefix, "start",  x0, np.exp(-x0), model(x0))
    print(prefix, "middle", x1, np.exp(-x1), model(x1))
    print(prefix, "end",    x2, np.exp(-x2), model(x2))
    print(prefix, "badness of the sampling:", np.std(sixf) / np.sqrt(2 ** log2N))
    return None

if __name__ == "__main__":
    np.random.seed(42)
    model = GaussianMolecule()
    Ps = model.get_Ps() # force construction of sampling
    for log2NK in [12, 16]:
        for iteration in range(20):
            for log2K in np.arange(0,9):
                make_and_fit_one_model(model, log2NK, log2K, iteration)
