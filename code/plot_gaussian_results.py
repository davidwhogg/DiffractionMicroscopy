"""
This file is part of the DiffractionMicroscopy project.
Copyright 2015 David W. Hogg (NYU).
"""

import glob
import numpy as np
import pickle as cp
import pylab as plt
from gaussian import *

Truth = 1. / np.array([47., 13., 11.]) # MUST BE ALIGNED WITH gaussian.py

def read_all_pickle_files():
    """
    Must be synchronized strictly with `gaussian.py`.
    """
    fns = glob.glob("./model_??_??.pkl")
    M = len(fns)
    Ns = np.zeros(M).astype(int)
    Ks = np.zeros(M).astype(int)
    ivars = np.zeros((M,3))
    sixfs = np.zeros((M,6))
    for i,fn in enumerate(fns):
        model, x0, x1, x2, sixf = read_pickle_file(fn)
        N, K = model.N, model.K
        Ns[i] = N
        Ks[i] = K
        ivars[i] = np.exp(x2)
        sixfs[i] = sixf
    return Ns, Ks, ivars, sixfs

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

def plot_posterior_sampling(datum, ivars, Ps):
    return None

def divergence(iv1, iv2):
    """
    Hard-coded to 3-d diagonals.
    """
    return 0.5 * (np.sum(iv1 / iv2) + np.sum(iv2 / iv1) - 6)

def plot_divergences(Ns, Ks, ivars):
    divs = np.array([divergence(ivar, Truth) for ivar in ivars])
    plt.clf()
    plt.scatter(Ns, Ks, c=np.log10(divs), s=200, cmap="gray")
    cb = plt.colorbar()
    cb.set_label("$\log_{10}$ divergence from Truth")
    plt.loglog()
    plt.xlim(np.min(Ns) / 2, np.max(Ns) * 2)
    plt.ylim(np.min(Ks) / 2, np.max(Ks) * 2)
    plt.xlabel("number of images $N$")
    plt.ylabel("number of photons per image $K$")
    plt.savefig("divergences.png")
    return None

def plot_sampling_badnesses(Ns, Ks, sixfs):
    badnesses = np.std(sixfs, axis=1) / np.sqrt(Ns) # per-image; hence sqrt(N)
    plt.clf()
    plt.scatter(Ns, Ks, c=np.log10(badnesses), s=200, cmap="gray")
    cb = plt.colorbar()
    cb.set_label("$\log_{10}$ per-image sampling badness")
    plt.loglog()
    plt.xlim(np.min(Ns) / 2, np.max(Ns) * 2)
    plt.ylim(np.min(Ks) / 2, np.max(Ks) * 2)
    plt.xlabel("number of images $N$")
    plt.ylabel("number of photons per image $K$")
    plt.savefig("badnesses.png")
    return None

if __name__ == "__main__":
    Ns, Ks, ivars, sixfs = read_all_pickle_files()
    plot_divergences(Ns, Ks, ivars)
    plot_sampling_badnesses(Ns, Ks, sixfs)
