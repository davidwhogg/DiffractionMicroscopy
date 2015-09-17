"""
This file is part of the DiffractionMicroscopy project.
Copyright 2015 David W. Hogg (NYU).

*Experiments with mixtures of Gaussians in the Fourier domain.*

# Bugs:
- hard-coded for two dimensions

"""
import numpy as np

lntwopi = np.log(2. * np.pi)

class MoG():

    def __init__(self, lnamps, means, vars):
        self.M = len(lnamps)
        self.set_lnamps(lnamps)
        self.set_means(means)
        self.set_vars(vars)
        self.check_internals()

    def set_lnamps(self, lnamps):
        assert lnamps.shape == (self.M, )
        self.lnamps = lnamps

    def get_lnamps(self):
        return self.lnamps

    def get_lnamp(self, m):
        return self.lnamps[m]

    def set_means(self, means):
        assert means.shape == (self.M, 2)
        self.means = means

    def get_means(self):
        return self.means

    def get_mean(self, m):
        return self.means[m]

    def _twod_determinant(self, tensor):
        """
        # Hard-coded two-d linear algebra.
        """
        return tensor[0, 0] * tensor[1, 1] - tensor[0, 1] * tensor[1, 0]

    def _twod_inverse_and_determinant(self, tensor):
        """
        # Hard-coded two-d linear algebra.
        """
        determinant = self._twod_determinant(tensor)
        inverse = np.array([[tensor[1, 1], -tensor[0, 1]], [-tensor[1, 0], tensor[0, 0]]])
        inverse /= determinant
        return inverse, determinant

    def set_vars(self, vars):
        assert vars.shape == (self.M, 2, 2)
        self.vars = vars
        self.ivars = np.zeros(vars.shape)
        self.dets = np.zeros(self.M)
        for m, var in enumerate(vars):
            ivar, det = self._twod_inverse_and_determinant(var)
            self.ivars[m, :, :] = ivar
            self.dets[m] = det

    def get_vars(self):
        return self.vars

    def get_var(self, m):
        return self.vars[m]

    def get_ivars(self):
        return self.ivars

    def get_ivar(self, m):
        return self.ivars[m]

    def get_dets(self):
        return self.dets

    def get_det(self, m):
        return self.dets[m]

    def check_internals(self):
        assert len(self.get_lnamps()) == self.M
        assert len(self.get_means()) == self.M
        assert len(self.get_vars()) == self.M
        for mean in self.get_means():
            assert mean.shape == (2, )
        for var in self.get_vars():
            assert var.shape == (2, 2)
            assert np.all(var == var.T)
            assert self._twod_determinant(var) > 0.

    def _ln_twod_gaussian(self, xx, mean, det, ivar):
        """
        # Hard-coded two-d Gaussian.
        """
        dx = xx - mean[None, :]
        print ((np.dot(dx, ivar) * dx).sum(axis=1)).shape
        return -lntwopi - 0.5 * np.log(det) - 0.5 * (np.dot(dx, ivar) * dx).sum(axis=1)

    def evaluate(self, x):
        xx = np.atleast_2d(x)
        value = np.zeros(len(x)).astype(float)
        for m in range(self.M):
            value += np.exp(self.get_lnamp(m) +
                            self._ln_twod_gaussian(xx, self.get_mean(m),
                                                   self.get_det(m), self.get_ivar(m)))
        return value

    def _ln_twod_FT_gaussian(self, xxi, mean, var):
        """
        # Hard-coded two-d Gaussian.
        
        ## bugs
        - Formula MADE UP; unchecked; note 2*pi**2
        """
        return -2. * np.pi * np.pi * (np.dot(xxi, var) * xxi).sum(axis=1) \
            - 2.j * np.pi * np.dot(xxi, mean)

    def evaluate_FT(self, xi):
        xxi = np.atleast_2d(xi)
        value = np.zeros(len(xi)).astype(complex)
        for m in range(self.M):
            value += np.exp(self.get_lnamp(m) +
                            self._ln_twod_FT_gaussian(xxi, self.get_mean(m),
                                                      self.get_var(m)))
        return value

if __name__ == "__main__":
    from matplotlib import pylab as plt
    tiny = 0.01
    xpix = np.arange(0. + 0.5 * tiny, 5., tiny)
    ypix = np.arange(-1. + 0.5 * tiny, 3., tiny)
    xv, yv = np.meshgrid(xpix, ypix)
    ny, nx = xv.shape
    x = np.vstack((xv.flatten(), yv.flatten())).T
    xi = x - np.array([2.5, 1.0])
    lnamps = np.array([0.1, 0.2, 0.2])
    means = np.array([[1., 0.], [2.0, 0.], [1.8, 1.0]])
    sigma = 0.2
    vars = np.array([[[sigma * sigma, 0.], [0., sigma * sigma]],
                     [[4. * sigma * sigma, 0.], [0., 4. * sigma * sigma]],
                     [[4. * sigma * sigma, 0.], [0., 4. * sigma * sigma]]])
    mog = MoG(lnamps, means, vars)
    f = mog.evaluate(x).reshape((ny, nx))
    ft = mog.evaluate_FT(xi).reshape((ny, nx))

    kwargs = {"cmap": "cool", "interpolation": "nearest", "origin": "lower"}
    plt.clf()
    plt.subplot(221)
    plt.imshow(f, **kwargs)
    plt.subplot(222)
    plt.imshow(ft.real, **kwargs)
    plt.subplot(223)
    plt.imshow(ft.imag, **kwargs)
    plt.subplot(224)
    plt.imshow((ft * ft.conj()).real, **kwargs)
    plt.savefig("images.png")
