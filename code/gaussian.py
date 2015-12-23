"""
This file is part of the DiffractionMicroscopy project.
Copyright 2015 David W. Hogg (NYU).
"""

import numpy as np
import pylab as plt

np.random.seed(42)
Truth = np.array([23., 13., 11.]) # axis-aligned variances

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

def make_fake_data(N=4096, K=128):
    """
    Every image contains exactly K photons.
    This is unrealistic, but suck it up.
    """
    Ps = make_random_projection_matrices(N)
    zero = np.zeros(2)
    samples = np.zeros((N, K, 2))
    for n in range(N):
        Vn = np.dot(Ps[n], Truth[:,None] * Ps[n].T)
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

def marginalized_likelihood(variances, data, Psampling):
    """
    Compute the sampling approximation to the marginalized likelihod.
    """
    return None

if __name__ == "__main__":
    foo = make_fake_data()
    show_data(foo, "foo")
