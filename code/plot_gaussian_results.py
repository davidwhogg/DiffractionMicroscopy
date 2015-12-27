"""
This file is part of the DiffractionMicroscopy project.
Copyright 2015 David W. Hogg (NYU).
"""

import numpy as np
import pickle as cp
import pylab as plt

Truth = 1. / np.array([47., 13., 11.]) # MUST BE ALIGNED WITH gaussian.py

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

def plot_divergences(Ns, Ks, ivarss):
    return None

if __name__ == "__main__":
    pass

