"""
This file is part of the PhaseRetrieval project.
Copyright 2015 David W. Hogg (SCDA) (NYU) (MPIA).
"""

import numpy as np
from numpy.fft import rfftn, irfftn

class pharetModel:

    def __init__(self, data, imageshape):
        """
        Must initialize the data, and the shape of the reconstructed image.
        """
        self.datashape = None
        self.imageshape = imageshape
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

    def __call__(self, image):
        self.set_image(image)
        return self.get_score()

if __name__ == "__main__":
    from matplotlib import pylab as plt
    from scipy.optimize import leastsq

    # make fake data
    np.random.seed(42)
    shape = (64, 64)
    trueimage = np.zeros(shape)
    yy, xx = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    for i in range(10):
        sigma2 = (np.random.uniform(1., 10.)) ** 2
        meanx, meany = np.random.uniform(0., shape[0]-1., size=2)
        foo = -0.5 * ((xx - meanx) ** 2 + (yy - meany) ** 2) / sigma2
        trueimage += np.exp(foo)
    for i in range(10):
        x1, x2, y1, y2 = np.random.uniform(0., shape[0]-1., size=4)
        trueimage[x1:x2, y1:y2] += 0.5
    trueft = rfftn(trueimage, shape)
    data = (trueft * trueft.conj()).real

    # make first plot
    plt.clf()
    plt.subplot(2,2,1)
    kwargs = {"interpolation": "nearest",
              "origin": "lower",
              "cmap": "afmhot"}
    plt.imshow(trueimage, **kwargs)
    plt.subplot(2,2,2)
    plt.imshow(np.log(data), **kwargs)
    plt.subplot(2,2,3)
    kwargs["cmap"] = "RdBu"
    foo = np.max((np.abs(min(np.min(trueft.real), np.min(trueft.imag))),
                  np.abs(max(np.max(trueft.real), np.max(trueft.imag)))))
    kwargs["vmin"] = -0.01 * foo
    kwargs["vmax"] = 0.01 * foo
    plt.imshow(trueft.real, **kwargs)
    plt.subplot(2,2,4)
    plt.imshow(trueft.imag, **kwargs)
    plt.savefig("whatev.png")

    # construct and test class
    model = pharetModel(data, shape)
    model.set_real_image(trueimage)
    print(model.get_score())
