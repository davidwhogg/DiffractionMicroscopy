"""
This file is part of the PhaseRetrieval project.
Copyright 2015 David W. Hogg (SCDA) (NYU) (MPIA).
"""

import numpy as np
from numpy.fft import rfftn, irfftn
from matplotlib import pylab as plt

class pharetModel:

    def __init__(self, data, imageshape, padding):
        """
        Must initialize the data, and the shape of the reconstructed image.
        """
        self.datashape = None
        self.imageshape = imageshape
        self.padding = padding
        self.set_data(data)

    def set_data(self, data):
        if self.datashape is None:
            self.datashape = data.shape
        assert self.datashape == data.shape
        self.data = data

    def set_real_image(self, image):
        assert self.imageshape == image.shape
        self.image = image
        self.ft = None

    def set_ft_image(self, ft):
        assert self.datashape == ft.shape
        self.ft = ft
        self.image = None
        
    def get_data(self):
        return self.data

    def get_real_image(self):
        if self.image is None:
            self.image = irfftn(self.ft, self.imageshape)
        return self.image

    def get_ft_image(self):
        if self.ft is None:
            self.ft = rfftn(self.image, self.imageshape)
        return self.ft

    def get_squared_norm_ft_image(self):
        ft = self.get_ft_image()
        return (ft * ft.conj()).real

    def get_data_residual(self):
        return self.get_squared_norm_ft_image() - self.get_data()

    def get_score(self):
        return np.sum(((self.get_data_residual()).real) ** 2)

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
        plt.title("{title}: score {s:.1f}".format(title=title, s=self.get_score()))
        kwargs["vmin"] = np.log(np.percentile(self.get_data(), 1.))
        kwargs["vmax"] = np.log(np.percentile(self.get_data(), 99.))
        plt.subplot(2,2,4)
        plt.imshow(np.log(self.get_data()), **kwargs)
        plt.title("data")
        plt.subplot(2,2,2)
        plt.title(title)
        plt.imshow(np.log(self.get_squared_norm_ft_image()), **kwargs)

    def set_real_image_from_vector(self, vector):
        """
        Note zero-padding insanity
        """
        pp = self.padding
        image = np.zeros(self.imageshape)
        image[pp:-pp,pp:-pp] = vector.reshape((self.imageshape[0] - 2 * pp,
                                               self.imageshape[1] - 2 * pp))
        self.set_real_image(image)

    def __call__(self, vector):
        self.set_real_image_from_vector(vector)
        return self.get_data_residual().flatten()

if __name__ == "__main__":
    from scipy.optimize import leastsq

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

    # construct and test class
    model = pharetModel(data, shape, padding)
    model.set_real_image(trueimage)
    print(model.get_score())

    # distort image
    guessimage = trueimage + 0.1 * np.random.normal(size=shape)
    guessimage = np.clip(guessimage, 0.01, np.Inf)[padding:-padding,padding:-padding]
    guessvector = guessimage.flatten()
    model.set_real_image_from_vector(guessvector)
    print(model.get_score())
    plt.clf()
    model.plot("before", truth=trueimage)
    plt.savefig("whatev1.png")

    # try optimization
    result = leastsq(model, guessvector)
    bettervector = result[0]
    print(model.get_score())
    plt.clf()
    model.plot("after 1", truth=trueimage)
    plt.savefig("whatev2.png")

    # try optimization again
    guessvector = bettervector.copy()
    result = leastsq(model, guessvector)
    bettervector = result[0]
    print(model.get_score())
    plt.clf()
    model.plot("after 2", truth=trueimage)
    plt.savefig("whatev3.png")
