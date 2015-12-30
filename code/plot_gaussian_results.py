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

def hogg_savefig(fn, **kwargs):
    print("writing file", fn)
    return plt.savefig(fn, **kwargs)

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
    models = []
    for i,fn in enumerate(fns):
        model, x0, x1, x2, sixf = read_pickle_file(fn)
        models.append(model)
        N, K = model.N, model.K
        Ns[i] = N
        Ks[i] = K
        ivars[i] = np.exp(x2)
        sixfs[i] = sixf
    return models, Ns, Ks, ivars, sixfs

def plot_datum(model, n):
    datum = model.get_datum(n)
    plt.plot(datum[:,0], datum[:,1], "k.", alpha=1./(model.K ** 0.25))
    return None

def plot_posterior_sampling(model, n):
    # compute lnlikes as in gaussian.py
    lnlikes = -0.5 * model.get_chisquareds(n) + 0.5 * model.K * model.get_lndets()
    # importance sample
    keep = np.random.uniform(size=len(lnlikes)) < np.exp(lnlikes - np.max(lnlikes))
    ivarts = model.get_ivarts()[keep]
    shortT, two, twoo = ivarts.shape
    assert two == 2
    assert twoo == 2
    ivartsample = ivarts[np.random.randint(shortT, size=16),:,:]
    for ivart in ivartsample:
        a, b = np.linalg.eigh(ivart)
        l1, l2 = 1. / np.sqrt(a) # sqrt because want deviation not variance
        v1, v2 = b[:,0], b[:,1]
        tiny = 0.001
        thetas = np.arange(0., 2. * np.pi + tiny, tiny)
        r = l1 * v1[None,:] * np.cos(thetas)[:,None] + l2 * v2[None,:] * np.sin(thetas)[:,None]
        r = r * 2. # just because!
        plt.plot(r[:,0], r[:,1], "r-", alpha=0.25)
    return None

def plot_data(model, sampling=False):
    log2N = (np.round(np.log2(model.N))+0.001).astype("int")
    log2K = (np.round(np.log2(model.K))+0.001).astype("int")
    prefix = "{:02d}_{:02d}".format(log2N, log2K)
    if sampling:
        prefix = "sampling_" + prefix
    else:
        prefix = "data_" + prefix
    plt.figure(figsize=(12,6))
    plt.clf()
    plt.subplots_adjust(bottom=0.06, top=0.94, left=0.06, right=0.94,
                        wspace=0.25, hspace=0.25)
    nex = np.min((18, model.N))
    for n in range(nex):
        plt.subplot(3, 6, n+1)
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plot_datum(model, n)
        if sampling:
            plot_posterior_sampling(model, n)
        if (n+1) != 13: # magic
            plt.gca().get_xaxis().set_ticklabels([])
            plt.gca().get_yaxis().set_ticklabels([])
        plt.axis("equal")
        plt.xlim(-20., 20)
        plt.ylim(plt.xlim())
        plt.title("image {}".format(n))
    return hogg_savefig(prefix + ".png")

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
    hogg_savefig("divergences.png")
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
    hogg_savefig("badnesses.png")
    return None

if __name__ == "__main__":
    np.random.seed(23)
    models, Ns, Ks, ivars, sixfs = read_all_pickle_files()

    for log2N, log2K in [(16, 0), (12, 4), (8, 8)]:
        thismodel = (np.where((Ns == 2 ** log2N) * (Ks == 2 ** log2K))[0])[0]
        model = models[thismodel]
        plot_data(model)
        plot_data(model, sampling=True)

    plot_divergences(Ns, Ks, ivars)
    plot_sampling_badnesses(Ns, Ks, sixfs)
