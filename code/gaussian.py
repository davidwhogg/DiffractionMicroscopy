"""
This file is part of the DiffractionMicroscopy project.
Copyright 2015 David W. Hogg (NYU).

## Bugs:
- Perhaps ought to cap the range on the ivars b/c runaway cases...?
- Need to check that T (number of samples) is large enough.
- I need to make the likelihood_one function pickleable and turn on multiprocessing.
- I ought to make the model a Class that is callable.
- I ought to cache (but carefully) previously called marginalized likelihoods b/c Powell sux.
-- The cache will get destroyed when the data are changed.
-- The cache will get destroyed when the orientation sampling is changed.

## Comments:
- I hard-coded some 2x2 linear algebra for speed.
"""

import numpy as np
import scipy.optimize as op
from scipy.misc import logsumexp
import pickle as cp
import pylab as plt

if False:
    from multiprocessing import Pool
    p = Pool(8)
    pmap = p.map
else:
    pmap = map

Truth = 1. / np.array([47., 13., 11.]) # axis-aligned inverse variances

def _normalize_vectors(vecs):
    return vecs / np.sqrt(np.sum(vecs * vecs, axis=1))[:,None]

def make_random_projection_matrices(N):
    xhat = _normalize_vectors(np.random.normal(size=(N, 3)))
    assert np.allclose(np.sum(xhat * xhat, axis = 1), 1.)
    yhat = np.random.normal(size=(N, 3))
    yhat = _normalize_vectors(yhat - xhat * np.sum(xhat * yhat, axis=1)[:,None])
    assert np.allclose(np.sum(xhat * yhat, axis = 1), 0.)
    assert np.allclose(np.sum(yhat * yhat, axis = 1), 1.)
    Ps = np.hstack((xhat, yhat)).reshape((N,2,3))
    return Ps

def make_projected_ivars(ivars, Ps):
    """
    Hard-coded for speed.
    """
    assert ivars.shape == (3, )
    T, two, three = Ps.shape
    assert two == 2
    assert three == 3
    ivarts = np.zeros((T, 2, 2))
    ivarts[:, 0, 0] = np.sum(Ps[:,0,:] * ivars[None,None,:] * Ps[:,0,:], axis=2)
    foo             = np.sum(Ps[:,0,:] * ivars[None,None,:] * Ps[:,1,:], axis=2)
    ivarts[:, 0, 1] = foo
    ivarts[:, 1, 0] = foo
    ivarts[:, 1, 1] = np.sum(Ps[:,1,:] * ivars[None,None,:] * Ps[:,1,:], axis=2)
    return ivarts

def get_2d_determinants(ivarts):
    """
    Hard-coded for speed.
    """
    T, two, twoo = ivarts.shape
    assert two == 2
    assert twoo == 2
    return ivarts[:,0,0] * ivarts[:,1,1] - ivarts[:,0,1] * ivarts[:,1,0]

def make_fake_data(N=2**15, K=2**4): # magic numbers
    """
    Every image contains exactly K photons.
    This is unrealistic, but suck it up.
    """
    Ps = make_random_projection_matrices(N)
    zero = np.zeros(2)
    samples = np.zeros((N, K, 2))
    ivarns = make_projected_ivars(Truth, Ps)
    for n in range(N):
        Vn = np.linalg.inv(ivarns[n])
        samples[n] = np.random.multivariate_normal(zero, Vn, size=K)
    return samples

def show_data(samples, prefix):
    plt.clf()
    nex = np.min((16, len(samples)))
    for n in range(nex):
        plt.subplot(4,4,n+1)
        plt.plot(samples[n,:,0], samples[n,:,1], "k.", alpha=0.5)
        plt.xlim(-15., 15.)
        plt.ylim(plt.xlim())
        plt.title("image {}".format(n))
    plt.savefig(prefix+".png")
    return None

def marginalized_ln_likelihood_one(datum, ivarts, logdets):
    """
    Compute the sampling approximation to the marginalized likelihood.
    """
    dd = np.atleast_2d(datum)
    chisquareds = np.sum(np.sum(np.tensordot(dd, ivarts, axes=(1,1)) * dd[:,None,:],
                                axis=2), axis=0) # should be length T
    loglikes = -0.5 * chisquareds + 0.5 * logdets # plus because INVERSE variances
    return logsumexp(loglikes)

def marginalized_ln_likelihood(ivars, data, Ps):
    """
    TOTALLY CONFUSED ABOUT WHAT map() DOES.
    """
    print("marginalized_ln_likelihood({}): starting...".format(1./ivars))
    assert ivars.shape == (3, )
    N, K, two = data.shape
    assert two == 2
    T, two, three = Ps.shape
    assert two == 2
    assert three == 3
    if np.any(ivars <= 0.):
        return -np.Inf
    ivarts = make_projected_ivars(ivars, Ps)
    logdets = K * np.log(get_2d_determinants(ivarts)) # factor of K for multiplicity
    foo = lambda x: marginalized_ln_likelihood_one(x, ivarts, logdets)
    lnlikes = np.array(list(pmap(foo, data)))
    print("marginalized_ln_likelihood({}): ...returning".format(1./ivars))
    return np.sum(lnlikes)

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

if __name__ == "__main__":

    np.random.seed(23)
    Ps = make_random_projection_matrices(1024)
    direc = np.array([[1., 1., 1.], [1., 0., -1.], [-1., 2., -1.]]) / 10.

    for log2NK in np.arange(9, 20):
        for log2K in np.arange(8, -1, -1):
            log2N = log2NK - log2K # MAGIC number of photons 2**19
            prefix = "{:02d}_{:02d}".format(log2N,log2K)
            print("starting run", prefix)

            # make fake data
            np.random.seed(42)
            data = make_fake_data(N=2**log2N, K=2**log2K)
            # show_data(data, "data_examples_"+prefix)

            # initialize empirically
            empvar = np.mean(data * data)
            x0 = np.log(1. / np.array([1.1 * empvar, empvar, 0.9 * empvar]))
            x0 = np.sort(x0)

            # optimize
            foo = lambda x: -2. * marginalized_ln_likelihood(np.exp(x), data, Ps)
            def bar(x): print(prefix, x, np.exp(-x))
            x1 = op.fmin_powell(foo, x0, callback=bar, direc=direc, xtol=1.e-3, ftol=1.e-5)
            x1 = np.sort(x1)
            x2 = op.fmin_powell(foo, x1, callback=bar, direc=direc, xtol=1.e-5, ftol=1.e-5)
            x2 = np.sort(x2)

            # save
            pickle_to_file("data_"+prefix+".pkl", (data, Ps, x0, x1, x2))
            print(prefix, "start",  x0, np.exp(-x0), foo(x0))
            print(prefix, "middle", x1, np.exp(-x1), foo(x1))
            print(prefix, "end",    x2, np.exp(-x2), foo(x1))
