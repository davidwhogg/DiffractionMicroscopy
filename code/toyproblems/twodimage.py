"""
This file is part of the DiffractionMicroscopy project.
Copyright 2016 David W. Hogg (NYU, SCDA).

This piece of code does nothing related to diffraction.
It only shows that you can reconstruct an image from small numbers of
photons taken in exoposures at unknown orientations.

# issues
- Should we apply the rotation projections to the sampling pixel
  points or to the Gaussian basis functions? Probably the latter.
"""
import pickle
import numpy as np


def hoggsumexp(qns, axis=None):
    """
    # purpose:
    - Computes `L = log(sum(exp(qns, axis=-1)))` but stably.
    - Also computes its N-dimensional gradient components dL / dg_m.

    # input
    - `qns`: ndarray of shape (n1, n2, n3, ..., nD, N)

    # output
    - `L`: ndarray of shape (n1, n2, n3, ..., nD)
    - `dL_dqns`: ndarray same shape as `qns`

    # issues
    - Not exhaustively tested.
    """
    if axis is None:
        axis = len(qns.shape) - 1
    Q = np.max(qns)
    expqns = np.exp(qns - Q)
    expL = np.sum(expqns, axis=axis)
    return np.log(expL) + Q, expqns / np.expand_dims(expL, axis)

class ImageModel:

    def __init__(self, ns, xnqs):
        self.N = int(np.max(ns)) + 1
        self.ns = ns
        self.xnqs = xnqs
        self.initialize_bases()
        self.create_angle_sampling()
        print("image_model:", self.lnams.shape, self.ns.shape, self.xms.shape,
              self.xnqs.shape)

    def initialize_bases(self):
        """
        Make the things you need for a grid of overlapping Gaussians.
        
        # issues
        - Magic numbers
        """
        self.sigma = 2.0  # magic
        self.sigma2 = self.sigma ** 2
        nyhalf, nxhalf = 12, 24  # magic
        yms, xms = np.meshgrid(np.arange(2 * nyhalf + 1),
                               np.arange(2 * nxhalf + 1))
        yms = (yms - nyhalf).flatten() * self.sigma  # lots of magic
        xms = (xms - nxhalf).flatten() * self.sigma
        self.M = len(yms)
        self.xms = np.vstack((yms, xms)).T
        self.lnams = np.random.normal(size=yms.shape)
        return None

    def create_angle_sampling(self, T=1024):  # MAGIC
        """
        # issues
        - Magic numbers.
        """
        self.T = T
        thetas = 2. * np.pi * np.random.uniform(size=self.T)
        self.costs = np.cos(thetas)
        self.sints = np.sin(thetas)
        return None

    def evaluate_lnbases(self, xtqs):
        """
        # input:
        - xtqs: ndarray of shape [T, Q, 2]

        # output:
        - lngtqms: evaluations of shape [T, Q, M]
        """
        T, Q, two = xtqs.shape
        assert two == 2
        return -0.5 * np.sum((xtqs[:, :, None, :] - self.xms[None, None, :, :]) ** 2, axis=3) / self.sigma2 \
            - np.log(2. * np.pi * self.sigma2)

    def pickle_to_file(self, fn):
        fd = open(fn, "wb")
        pickle.dump(self, fd)
        fd.close()
        print(fn)
        return None

    def plot(self, ax):
        """
        Put a two-d image onto a matplotlib plot.

        # issues:
        - Magic numbers.
        - Requires matplotlib (or the ducktype).
        """
        ys = np.arange(-30.5, 31, 1) # magic
        xs = np.arange(-50.5, 51, 1) # magic
        ys, xs = np.meshgrid(ys, xs)
        ny, nx = ys.shape
        xps = np.zeros((ny, nx, 2))
        xps[:, :, 0] = ys
        xps[:, :, 1] = xs
        image = np.sum(np.exp(self.lnams[None, None, :] + self.evaluate_lnbases(xps)), axis=2) # unsafe
        vmin = -0.75 * np.max(image)
        ax.imshow(-image.T, interpolation="nearest", origin="lower", vmin=vmin, vmax=0.)
        return None

    def rotate(self, xqs):
        """
        # input:
        - xqs: ndarray of shape [Q, 2]

        # output:
        - xtqs: ndarray of shape [self.T, Q, 2]
        """
        xtqs = self.costs[:, None, None] * xqs[None, :, :]
        xtqs[:, :, 0] += self.sints[:, None] * xqs[None, :, 1]
        xtqs[:, :, 1] -= self.sints[:, None] * xqs[None, :, 0]
        return xtqs

    def single_image_lnlike(self, n):
        """
        # input:
        - n: index of the image for which lnL should be computed

        # output:
        - lnLn, dlnLn_dlnams: lnL and its gradient wrt self.lnams

        # issues:
        - Not tested.
        - Too many asserts!
        """
        I = (self.ns == n)
        Q = np.sum(I)
        xqs = (self.xnqs[I]).reshape((Q, 2))
        xtqs = self.rotate(xqs)
        assert xtqs.shape == (self.T, Q, 2)
        lngtqms = self.evaluate_lnbases(xtqs)
        assert lngtqms.shape == (self.T, Q, self.M)
        # logsumexp over m index
        lnLntqs, dlnLntqs_dlnams = hoggsumexp(self.lnams[None, None, :] + lngtqms, axis=2)
        assert lnLntqs.shape == (self.T, Q)
        assert dlnLntqs_dlnams.shape == (self.T, Q, self.M)
        # sum over q index
        lnLnts = np.sum(lnLntqs, axis=1)
        assert lnLnts.shape == (self.T, )
        dlnLnts_dlnams = np.sum(dlnLntqs_dlnams, axis=1)
        assert dlnLnts_dlnams.shape == (self.T, self.M)
        # logsumexp over t index
        lnLn, dlnLn_dlnLnts = hoggsumexp(lnLnts, axis=0)
        assert dlnLn_dlnLnts.shape == (self.T, )
        dlnLn_dlnams = np.sum(dlnLn_dlnLnts[:, None] * dlnLnts_dlnams, axis=0)
        assert dlnLn_dlnams.shape == (self.M, )
        dpenalty_dlnams = np.exp(self.lnams)
        penalty = np.sum(dpenalty_dlnams)
        return lnLn - penalty, dlnLn_dlnams - dpenalty_dlnams


def test_hoggsumexp():
    for shape in [(7, ), (3, 5, 9)]:
        qns = np.random.normal(size=shape)
        L, dL = hoggsumexp(qns)
        if len(shape) == 3:
            assert L.shape == (3, 5)
            assert dL.shape == (3, 5, 9)
        delta = 1e-5
        if len(shape) == 1:
            qns[3] += delta
        else:
            qns[2, 2, 4] += delta
        L1, foo = hoggsumexp(qns)
        if len(shape) == 1:
            qns[3] -= 2. * delta
        else:
            qns[2, 2, 4] -= 2. * delta
        L2, foo = hoggsumexp(qns)
        if len(shape) == 1:
            print("test_hoggsumexp():", dL[3], (L1 - L2) / (2. * delta))
        else:
            print("test_hoggsumexp():", dL[2, 2, 4], (L1 - L2) / (2. * delta))
    return True

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ns, xnqs, rnd_state = np.load('./photons.npy')
    np.random.set_state(rnd_state)

    # initialize model
    model = ImageModel(ns, xnqs)

    # check derivative
    ##Ln, gradLn = model.single_image_lnlike(0)
    ##delta = 1.e-5 # magic
    ##model.lnams[5] += delta
    ##Ln2, gradLn2 = model.single_image_lnlike(0)
    ##print(gradLn[5], (Ln2 - Ln) / delta)

    # take a few gradient steps
    fig = plt.figure()
    sumh = 0.
    hplot = 0.
    for j in range(2 ** 16):
        h = 4096. / (2048. + float(j)) # magic
        sumh += h
        n = np.random.randint(model.N)
        Ln, gradLn = model.single_image_lnlike(n)
        print("stochastic", j, h, n, Ln)
        model.lnams += h * gradLn    

        # plot the output of the s.g.
        if sumh > hplot:
            hplot += 10.
            pfn = "./model_{:06d}.pkl".format(j)
            model.pickle_to_file(pfn)
            plt.clf()
            plt.gray()
            ax = plt.gca()
            model.plot(ax)
            pfn = "./model_{:06d}.png".format(j)
            plt.savefig(pfn)
            print(pfn)
