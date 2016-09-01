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
        - Magic numbers.
        - The three-d model is actually two-d, which is cheating!!
        """
        self.sigma = 3.0  # magic # 2.0
        self.sigma2 = self.sigma ** 2
        nyhalf, nxhalf = 16, 16  # magic # 12, 24
        yms, xms = np.meshgrid(np.arange(2 * nyhalf + 1),
                               np.arange(2 * nxhalf + 1))
        yms = (yms - nyhalf).flatten() * self.sigma  # lots of magic
        xms = (xms - nxhalf).flatten() * self.sigma
        zms = np.zeros_like(yms) # this is cheating!!
        self.M = len(yms)
        self.xms = np.vstack((yms, xms, zms)).T
        self.lnams = np.random.normal(size=yms.shape)
        return None

    def create_angle_sampling(self, T=2**10):  # MAGIC 1024
        """
        # issues
        - Ought to re-draw yhats that have large dot products with xhats...
        """
        self.T = T
        self.rotations = np.zeros((T, 2, 3))
        xhats = np.random.normal(size=3*T).reshape((T, 3))
        xhats /= np.sqrt(np.sum(xhats**2, axis=1))[:, None]
        yhats = np.random.normal(size=3*T).reshape((T, 3))
        yhats -= np.sum(xhats*yhats, axis=1)[:,None] * xhats
        yhats /= np.sqrt(np.sum(yhats**2, axis=1))[:, None]
        self.rotations[:, 0, :] = xhats
        self.rotations[:, 1, :] = yhats
        return None

    def evaluate_rotated_lnbases(self, xqs):
        """
        # input:
        - xqs: ndarray of shape [Q, 2]

        # output:
        - lngtqms: evaluations of shape [T, Q, M]
        """
        Q, two = xqs.shape
        assert two == 2
        
        xtms = np.sum(self.rotations[:, None, :, :] * self.xms[None, :, None, :], axis=3)
        print(xqs.shape, xtms.shape)
        return -0.5 * np.sum((xqs[None, :, None, :] - xtms[:, None, :, :]) ** 2, axis=3) / self.sigma2 \
            - np.log(2. * np.pi * self.sigma2)

    def evaluate_lnbases(self, xfqs):
        """
        # input:
        - xfqs: ndarray of shape [foo, Q, 2]

        # output:
        - lngfqms: evaluations of shape [foo, Q, M]
        """
        return -0.5 * np.sum((xfqs[:, :, None, :] - self.xms[None, None, :, :2]) ** 2, axis=3) / self.sigma2 \
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
        f = 0.65 # magic
        ys = np.arange(-self.sigma * f * np.sqrt(self.M), self.sigma * f * np.sqrt(self.M), 1) # magic
        xs = np.arange(-self.sigma * f * np.sqrt(self.M), self.sigma * f * np.sqrt(self.M), 1) # magic
        ys, xs = np.meshgrid(ys, xs)
        ny, nx = ys.shape
        xps = np.zeros((ny, nx, 2))
        xps[:, :, 0] = ys
        xps[:, :, 1] = xs
        image = np.sum(np.exp(self.lnams[None, None, :] + self.evaluate_lnbases(xps)), axis=2) # unsafe
        vmin = -0.75 * np.max(image)
        ax.imshow(-image.T, interpolation="nearest", origin="lower", vmin=vmin, vmax=0.)
        return None

    def single_image_lnlike(self, n):
        """
        # input:
        - n: index of the image for which lnL should be computed

        # output:
        - lnLn, dlnLn_dlnams: lnL and its gradient wrt self.lnams

        # issues:
        - Not tested.
        - Too many asserts!
        - Is the penalty and its derivative correct? Hogg is suspicious.
        """
        I = (self.ns == n)
        Q = np.sum(I)
        xqs = (self.xnqs[I]).reshape((Q, 2))
        lngtqms = self.evaluate_rotated_lnbases(xqs)
        assert lngtqms.shape == (self.T, Q, self.M)
        # logsumexp over m index (ie, summing the mixture of Gaussians)
        lnLntqs, dlnLntqs_dlnams = hoggsumexp(self.lnams[None, None, :] + lngtqms, axis=2)
        assert lnLntqs.shape == (self.T, Q)
        assert dlnLntqs_dlnams.shape == (self.T, Q, self.M)
        # sum over q index (ie, product together all the photons in image n)
        lnLnts = np.sum(lnLntqs, axis=1)
        assert lnLnts.shape == (self.T, )
        dlnLnts_dlnams = np.sum(dlnLntqs_dlnams, axis=1)
        assert dlnLnts_dlnams.shape == (self.T, self.M)
        # logsumexp over t index (ie, marginalize out the angles)
        lnLn, dlnLn_dlnLnts = hoggsumexp(lnLnts, axis=0)
        assert dlnLn_dlnLnts.shape == (self.T, )
        dlnLn_dlnams = np.sum(dlnLn_dlnLnts[:, None] * dlnLnts_dlnams, axis=0)
        assert dlnLn_dlnams.shape == (self.M, )
        dpenalty_dlnams = np.exp(self.lnams)
        penalty = np.sum(dpenalty_dlnams) # is this correct?
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

    photons = np.load('./photons.npy')
    ns = photons[:, 0]
    xnqs = photons[:, 1:]

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
            hplot += 40.
            pfn = "./model_{:06d}.pkl".format(j)
            model.pickle_to_file(pfn)
            plt.clf()
            plt.gray()
            ax = plt.gca()
            model.plot(ax)
            pfn = "./model_{:06d}.png".format(j)
            plt.savefig(pfn)
            print(pfn)
