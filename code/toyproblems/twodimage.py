"""
This file is part of the DiffractionMicroscopy project.
Copyright 2016 David W. Hogg (NYU, SCDA).

This piece of code does nothing related to diffraction.
It only shows that you can reconstruct an image from small numbers of
photons taken in exoposures at unknown orientations.
"""
import numpy as np

def hoggsumexp(qns, dqn_dams, diag=False):
    """
    # purpose:
    - Computes L = log(sum(exp(qns, axis=-1))).
    - Also computes its M-dimensional gradient components dL / da_m.

    # input
    - qns: ndarray of shape [n1, n2, n3, ..., nD, N]
    - dqn_dams: ndarray of shape [n1, n2, n3, ..., nD, N, M]
    - diag: if True, then dqn_dams.shape == dqn_dams.shape and [read the source]

    # output
    - L: ndarray of shape [n1, n2, n3, ..., nD]
    - dL_dams: ndarray of shape [n1, n2, n3, ..., nD, M]

    # issues
    - Not exhaustively tested.
    """
    axis = len(qns.shape) - 1
    if diag:
        assert qns.shape == dqn_dams.shape
    Q = np.max(qns)
    expqns = np.exp(qns - Q)
    expL = np.sum(expqns, axis=axis)
    if diag:
        numerator = expqns * dqn_dams
    else:
        numerator = np.sum(np.expand_dims(expqns, axis + 1) * dqn_dams, axis=axis)
    return np.log(expL) + Q, numerator / np.expand_dims(expL, axis)

class image_model:

    def __init__(self):
        initialize_bases()
        create_angle_sampling()
        return self

    def initialize_bases(self):
        self.M = whatevs
        self.various_things = whatevs
        self.lnams = whatevs
        return None

    def create_angle_sampling(self):
        self.T = 1024 # MAGIC
        self.costs = np.cos(self.thetas)
        self.sints = np.sin(self.thetas)
        return None

    def evaluate_lnbases(self, xtqs):
        """
        # input:
        - xtqs: ndarray of shape [T, Q, 2]

        # output:
        - lngtqms: evaluations of shape [T, Q, M]
        """
        return lngtqms

    def rotate(xqs):
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

    def single_image_lnlike(n):
        """
        # input:
        - n: index of the image for which lnL should be computed

        # output:
        - lnLn, dlnLn_dlnams: lnL and its gradient wrt self.lnams
        """
        I = (self.ns == n)
        Q = np.sum(I)
        xqs = (self.xs[I]).reshape((Q, 2))
        xtqs = self.rotate(xqs)
        assert xtqs.shape == (self.T, Q, 2)
        lngtqms = self.evaluate_lnbases(self, xtqs)
        lnLntqs = logsumexp(self.lnams[None, None, :] + lngtqms, axis=2)
        dlnLntqs_dlnams = whatevs
        lnLnts = np.sum(lnLntqs, axis=1)
        dlnLnts_dlnams = np.sum(dlnLntqs_dlnams, axis=1)
        lnLn = logsumexp(lnLnts)
        dlnLn_dlnams = whatevs
        return lnLn, dlnLn_dlnams

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
        dns = np.ones_like(qns)
        L, dL = hoggsumexp(qns, dns, diag=True)
        if len(shape) == 3:
            assert L.shape == (3, 5)
            assert dL.shape == (3, 5, 9)
        delta = 1e-5
        if len(shape) == 1:
            qns[3] += delta
        else:
            qns[2, 2, 4] += delta
        L1, foo = hoggsumexp(qns, dns, diag=True)
        if len(shape) == 1:
            qns[3] -= 2. * delta
        else:
            qns[2, 2, 4] -= 2. * delta
        L2, foo = hoggsumexp(qns, dns, diag=True)
        if len(shape) == 1:
            print(dL[3], (L1 - L2) / (2. * delta))
        else:
            print(dL[2, 2, 4], (L1 - L2) / (2. * delta))

        if len(shape) == 1:
            qns[3] += delta # restore
            dns = np.eye(shape[-1])[:,0:4]
            L3, dL3 = hoggsumexp(qns, dns)
            print(L, L3, dL[3], dL3[3])
    return True

if __name__ == "__main__":
    import pylab as plt
    np.random.seed(42)

    test_hoggsumexp()
    assert False

    truth = make_truth()

    ns, ys, xs = make_fake_data(truth)
    print(ns.shape, ys.shape, xs.shape)

    plt.figure()
    plt.clf()
    plt.plot(xs, ys, 'k.', alpha=0.5)
    plt.axis("equal")
    plt.savefig("./test.png")
