"""
This file is part of the DiffractionMicroscopy project.
Copyright 2015 David W. Hogg (NYU).

## Comments
- I have hard-coded some 2x2 linear algebra for speed.
"""

import numpy as np
import pylab as plt

np.random.seed(42)
Truth = np.array([1./47., 1./13., 1./11.]) # axis-aligned inverse variances

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
    ivarns = np.zeros((T, 2, 2))
    ivarns[:, 0, 0] = np.sum(Ps[:,0,:] * ivars[None,None,:] * Ps[:,0,:], axis=2)
    foo             = np.sum(Ps[:,0,:] * ivars[None,None,:] * Ps[:,1,:], axis=2)
    ivarns[:, 0, 1] = foo
    ivarns[:, 1, 0] = foo
    ivarns[:, 1, 1] = np.sum(Ps[:,1,:] * ivars[None,None,:] * Ps[:,1,:], axis=2)
    return ivarns

def get_2d_determinants(ivarns):
    """
    Hard-coded for speed.
    """
    T, two, twoo = ivarns.shape
    assert two == 2
    assert twoo == 2
    return ivarns[:,0,0] * ivarns[:,1,1] - ivarns[:,0,1] * ivarns[:,1,0]

def make_fake_data(N=4096, K=128):
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
    for n in range(16):
        plt.subplot(4,4,n+1)
        plt.plot(samples[n,:,0], samples[n,:,1], "k.", alpha=0.5)
        plt.xlim(-15., 15.)
        plt.ylim(plt.xlim())
        plt.title("image {}".format(n))
    plt.savefig(prefix+".png")
    return None

def marginalized_ln_likelihood_one(ivars, datum, Ps):
    """
    Compute the sampling approximation to the marginalized likelihood.

    NOT YET WRITTEN
    """
    assert ivars.shape == (3, )
    K, two = datum.shape
    assert two == 2
    T, two, three = Ps.shape
    assert two == 2
    assert three == 3
    threevectors = np.tensordot(datum, Ps, axes=(1,1))
    print(threevectors.shape)
    print(K, T)
    assert False
    return len(datum)

def marginalized_ln_likelihood(ivars, data, Ps):
    """
    TOTALLY CONFUSED ABOUT WHAT map() DOES.
    """
    foo = lambda x: marginalized_ln_likelihood_one(ivars, x, Ps)
    lnlikes = np.array(list(map(foo, data)))
    return np.sum(lnlikes)

if __name__ == "__main__":
    data = make_fake_data()
    show_data(data, "data_examples")
    ivars = 1. / np.array([20.,19.,18.])
    Ps = make_random_projection_matrices(1024)
    foo = marginalized_ln_likelihood(ivars, data, Ps)
    print(foo)
