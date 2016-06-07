"""
This file is part of the DiffractionMicroscopy project.
Copyright 2016 David W. Hogg (NYU, SCDA).

This piece of code does nothing related to diffraction.
It only shows that you can reconstruct an image from small numbers of
photons taken in exoposures at unknown orientations.

"""
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

class image_model:

    def __init__(self, ns, xnqs):
        self.N = int(np.max(ns)) + 1
        self.ns = ns
        self.xnqs = xnqs
        self.initialize_bases()
        self.create_angle_sampling()
        print(self.lnams.shape, self.ns.shape, self.xms.shape, self.xnqs.shape)
        return None

    def initialize_bases(self):
        """
        - Make the things you need for a grid of overlapping Gaussians.
        """
        self.sigma = 2. # magic
        self.sigma2 = self.sigma ** 2
        nyhalf, nxhalf = 7, 14 # magic
        yms, xms = np.meshgrid(np.arange(2 * nyhalf + 1), np.arange(2 * nxhalf + 1))
        yms = (yms - nyhalf).flatten() * self.sigma # lots of magic
        xms = (xms - nxhalf).flatten() * self.sigma
        self.M = len(yms)
        self.xms = np.vstack((yms, xms)).T
        self.lnams = np.zeros_like(yms)
        return None

    def create_angle_sampling(self):
        self.T = 1024 # MAGIC
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
        assert T == self.T
        assert two == 2
        return -0.5 * np.sum((xtqs[:, :, None, :] - self.xms[None, None, :, :]) ** 2, axis=3) / self.sigma2 \
            - np.log(2. * np.pi * self.sigma2)

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

def make_truth():
    """
    OMG dumb.
    dependency: PyPNG
    """
    import pylab as plt
    import png
    plt.figure(figsize=(0.5,0.3))
    plt.clf()
    plt.text(0.5, 0.5, r"Vera",
             ha="center", va="center",
             clip_on=False,
             transform=plt.gcf().transFigure);
    plt.gca().set_axis_off()
    datafn = "./vera.png"
    plt.savefig(datafn, dpi=200)
    w, h, pixels, metadata = png.Reader(filename=datafn).read_flat()
    pixels = (np.array(pixels).reshape((h,w,4)))[::-1,:,0]
    pixels = (np.max(pixels) - pixels) / np.max(pixels)
    print(w, h, pixels.shape, metadata)
    return pixels

def get_one_photon(image):
    """
    stupidly uses rejection sampling!
    lots of annoying details.
    """
    h, w = image.shape
    maxi = np.max(image)
    count = 0
    while(count == 0):
        yy = np.random.randint(h)
        xx = np.random.randint(w)
        if (image[yy, xx] > maxi * np.random.uniform()):
            count = 1
    return yy - h / 2 + np.random.uniform(), xx - w / 2 + np.random.uniform()

def make_fake_image(truth, Q):
    """
    # inputs:
    - truth: pixelized image of density
    - Q: number of photons to make
    """
    theta = 2. * np.pi * np.random.uniform()
    ct, st = np.cos(theta), np.sin(theta)
    ys = np.zeros(Q)
    xs = np.zeros(Q)
    for q in range(Q):
        yy, xx = get_one_photon(truth)
        xs[q] = ct * xx + st * yy
        ys[q] = -st * xx + ct * yy
    return ys, xs

def make_fake_data(truth, N=1024, rate=1.):
    """
    # inputs:
    - truth: pixelized image of density
    - N: number of images to take
    - rate: mean number of photons per image

    # notes:
    - Images that get zero photons will be dropped, but N images will be returned.
    """
    ns, ys, xs = [], [], []
    for n in range(N):
        Q = 0
        while(Q == 0):
            Q = np.random.poisson(rate)
        tys, txs = make_fake_image(truth, Q)
        for q in range(Q):
            ns.append(n)
            ys.append(tys[q])
            xs.append(txs[q])
    return np.array(ns), np.array(ys), np.array(xs)

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
            print(dL[3], (L1 - L2) / (2. * delta))
        else:
            print(dL[2, 2, 4], (L1 - L2) / (2. * delta))
    return True

if __name__ == "__main__":
    import pylab as plt
    np.random.seed(42)

    # make fake data
    truth = make_truth()
    ns, ys, xs = make_fake_data(truth, N=16, rate=16.)
    xnqs = np.vstack((ys, xs)).T
    print(ns.shape, ys.shape, xs.shape, xnqs.shape)

    # plot fake data
    plt.figure()
    plt.clf()
    plt.plot(xs, ys, 'k.', alpha=0.5)
    plt.axis("equal")
    plt.savefig("./test.png")

    # initialize model
    model = image_model(ns, xnqs)
    Ln, gradLn = model.single_image_lnlike(0)
    print(Ln, gradLn)

    # check derivative
    ##delta = 1.e-5 # magic
    ##model.lnams[5] += delta
    ##Ln2, gradLn2 = model.single_image_lnlike(0)
    ##print(gradLn[5], (Ln2 - Ln) / delta)

    # take a few gradient steps
    h = 0.1 # magic
    for j in range(128):
        n = np.random.randint(model.N)
        Ln, gradLn = model.single_image_lnlike(n)
        print("stochastic", j, n, Ln)
        model.lnams += h * gradLn    

