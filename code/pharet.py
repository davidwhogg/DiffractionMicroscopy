"""
This file is part of the PhaseRetrieval project.
Copyright 2015 David W. Hogg (SCDA) (NYU) (MPIA).
"""

import numpy as np
from numpy.fft import rfftn, irfftn, fftshift, ifftshift
from matplotlib import pylab as plt
import emcee

class pharetModel:

    def __init__(self, data, imageshape, padding, ivar=None):
        """
        Must initialize the data, and the shape of the reconstructed image.
        """
        self.datashape = None
        self.imageshape = imageshape
        self.padding = padding
        self.set_data(data)
        if ivar is not None:
            self.set_ivar(ivar)

    def set_data(self, data):
        if self.datashape is None:
            self.datashape = data.shape
        assert self.datashape == data.shape
        self.data = data

    def set_ivar(self, ivar):
        assert self.datashape == ivar.shape
        self.ivar = ivar

    def set_real_image(self, image):
        assert self.imageshape == image.shape
        self.image = image
        self.ft = None

    def set_real_image_from_vector(self, vector):
        """
        Note zero-padding insanity
        """
        pp = self.padding
        image = np.zeros(self.imageshape)
        image[pp:-pp,pp:-pp] = np.exp(vector).reshape((self.imageshape[0] - 2 * pp,
                                                       self.imageshape[1] - 2 * pp))
        self.set_real_image(image)

    def set_ft_image(self, ft):
        assert self.datashape == ft.shape
        self.ft = ft
        self.image = None
        
    def get_data(self):
        return self.data

    def get_ivar(self):
        return self.ivar

    def get_real_image(self):
        if self.image is None:
            self.image = irfftn(self.ft, self.imageshape)
        return self.image

    def get_real_image_vector(self):
        """
        Note zero-padding insanity
        """
        pp = self.padding
        return np.log(self.get_real_image()[pp:-pp,pp:-pp]).flatten()

    def get_ft_image(self):
        if self.ft is None:
            self.ft = rfftn(self.image, self.imageshape)
        return self.ft

    def get_squared_norm_ft_image(self):
        ft = self.get_ft_image()
        return (ft * ft.conj()).real

    def get_data_residual(self):
        return self.get_squared_norm_ft_image() - self.get_data()

    def get_gaussian_ln_like(self):
        return -0.5 * np.sum(self.ivar * self.get_data_residual() ** 2)

    def get_score_L1(self):
        return np.sum(np.abs(self.get_data_residual()))

    def get_score_L2(self):
        return np.sum(((self.get_data_residual()).real) ** 2)

    def do_one_crazy_map(self, tiny=1.e-10):
        """
        Do one iteration of the solution map of Magland et al.
        """
        oldimage = self.get_real_image().copy()
        oldft = self.get_ft_image().copy()
        # fix squared norm
        newft = oldft * np.sqrt(self.get_data() / self.get_squared_norm_ft_image())
        self.set_ft_image(0.4 * newft + 0.6 * oldft)
        # zero out borders
        newimage = self.get_real_image().copy()
        pp = self.padding
        newimage[:pp,:] = 0.
        newimage[:,:pp] = 0.
        newimage[-pp:,:] = 0.
        newimage[:,-pp:] = 0.
        # clip negatives
        newimage = np.clip(newimage, tiny * np.max(newimage), np.Inf)
        self.set_real_image(2. * newimage / 3. + oldimage / 3.)

    def get_ks(self):
        """
        Get k magnitude for every fft / data pixel.

        ## bugs:
        - brittle and untested
        """
        n0 = self.datashape[0]
        x0 = ifftshift(np.arange(n0) - np.floor(n0 / 2))
        x = []
        for nd in self.datashape[-1:0:-1]:
            x.append(np.arange(nd))
        x.append(x0)
        return np.sqrt(np.sum(np.array(np.meshgrid(*x)) ** 2, axis=0))

    def plot(self, title, truth=None):
        kwargs = {"interpolation": "nearest",
                  "origin": "lower",
                  "cmap": "afmhot",
                  "vmin": 0.0,
                  "vmax": np.max(self.get_real_image())}
        if truth is not None:
            plt.subplot(2,2,3)
            plt.imshow(truth, **kwargs)
            plt.title("truth")
        plt.subplot(2,2,1)
        plt.imshow(self.get_real_image(), **kwargs)
        plt.title("{title}: scores {s1:.1f} {s2:.1f}".format(title=title,
                                                             s1=self.get_score_L1(),
                                                             s2=self.get_score_L2()))
        kwargs["vmin"] = np.log(np.percentile(self.get_data(), 1.))
        kwargs["vmax"] = np.log(np.percentile(self.get_data(), 99.))
        plt.subplot(2,2,4)
        data = np.log(self.get_data().copy())
        data[np.where(self.get_ivar() <= 0)] = kwargs["vmin"]
        plt.imshow(fftshift(data, axes=0), **kwargs)
        plt.title("data")
        plt.subplot(2,2,2)
        plt.title(title)
        plt.imshow(np.log(fftshift(self.get_squared_norm_ft_image(), axes=0)), **kwargs)

    def __call__(self, vector, output):
        self.set_real_image_from_vector(vector)
        if output == "lnprob":
            return self.get_gaussian_ln_like()
        if output == "resid":
            return self.get_data_residual().flatten()
        if output == "L1":
            return self.get_score_L1()
        if output == "L2":
            return self.get_score_L2()
        assert False

if __name__ == "__main__":
    from scipy.optimize import leastsq, minimize

    # make fake data
    np.random.seed(42)
    shape = (64, 64)
    padding = 16
    trueimage = np.zeros(shape)
    yy, xx = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    for i in range(10):
        sigma2 = (np.random.uniform(0.5, 5.)) ** 2
        meanx, meany = np.random.uniform(padding+2., shape[0]-padding-2., size=2)
        foo = -0.5 * ((xx - meanx) ** 2 + (yy - meany) ** 2) / sigma2
        trueimage += np.exp(foo)
    for i in range(10):
        x1, y1 = np.random.uniform(padding+1., shape[0]-padding-7., size=2)
        dx1, dy1 = np.random.uniform(1., 6., size=2)
        trueimage[y1:y1+dy1, x1:x1+dx1] += 0.5
    trueimage[:padding,:] = 0.
    trueimage[:,:padding] = 0.
    trueimage[-padding:,:] = 0.
    trueimage[:,-padding:] = 0.
    trueft = rfftn(trueimage, shape)
    data = (trueft * trueft.conj()).real

    # construct an inverse variance "noise level"
    sigma = np.zeros_like(data) + 0.05 * np.median(data)
    sigma2 += sigma ** 2 + (0.05 * data) ** 2
    ivar = 1. / sigma2

    # construct and test class
    model = pharetModel(data, shape, padding, ivar=ivar)

    # initialize emcee
    ndim = 32 * 32
    nwalkers = 2 * ndim + 2
    pos = np.random.normal(size=(nwalkers, ndim))
    sampler = emcee.EnsembleSampler(nwalkers, ndim, model, args=["lnprob", ])
    jj = 0
    for k in np.arange(64) + 0.1:
        sampler.reset()
        thisivar = ivar.copy()
        thisivar[np.where(model.get_ks() > k)] = 0.
        model.set_ivar(thisivar)
        pos, prob, state = sampler.run_mcmc(pos, 128)
        print("{1:.2f} Mean acceptance fraction: {0:.3f}"
              .format(np.mean(sampler.acceptance_fraction), k))
        for mm in range(3):
            model.set_real_image_from_vector(pos[mm])
            plt.clf()
            model.plot("before", truth=trueimage)
            plt.savefig("whatev{jj:03d}.png".format(jj=jj))
            jj += 1

if False:
    model.set_real_image(trueimage)
    print(model.get_score_L1(), model.get_score_L2())

    # distort image
    guessimage = trueimage + 0.1 * np.random.normal(size=shape)
    guessimage = np.clip(guessimage, 0.01, np.Inf)[padding:-padding,padding:-padding]
    guessvector = np.log(guessimage.flatten())
    model.set_real_image_from_vector(guessvector)
    jj = 0
    print(jj, model.get_score_L1(), model.get_score_L2())
    plt.clf()
    model.plot("before", truth=trueimage)
    plt.savefig("whatev{jj:1d}.png".format(jj=jj))

    # try optimization by a schedule of minimizers
    method = "Powell"
    maxfev = 100000
    bettervector = guessvector.copy()
    for ii in range(10):
        jj = ii + 1

        if ii == 0:
            # zeroth crazy map
            guessvector = bettervector.copy()
            model.set_real_image_from_vector(guessvector)
            for qq in range(1, 2 ** 16 + 1):
                model.do_one_crazy_map()
                if qq > 1000 and qq == 2 ** np.floor(np.log(qq) / np.log(2)).astype(int):
                    print(jj, qq, model.get_score_L1(), model.get_score_L2())
            bettervector = model.get_real_image_vector()

        # first levmar
        guessvector = bettervector.copy()
        result = leastsq(model, guessvector, args=("resid", ), maxfev=maxfev)
        bettervector = result[0]
        model.set_real_image_from_vector(bettervector)
        print(jj, model.get_score_L1(), model.get_score_L2())

        # second L1 minimization
        guessvector = bettervector.copy()
        result = minimize(model, guessvector, args=("L1", ), method=method,
                          options={"maxfev" : maxfev})
        bettervector = result["x"]
        model.set_real_image_from_vector(bettervector)
        print(jj, model.get_score_L1(), model.get_score_L2())

        # make plots
        plt.clf()
        model.plot("after {jj:02d}".format(jj=jj), truth=trueimage)
        plt.savefig("whatev{jj:02d}.png".format(jj=jj))
