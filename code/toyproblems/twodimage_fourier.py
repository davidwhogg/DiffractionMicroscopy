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

try:
    import numexpr as ne
    ne.set_num_threads(ne.ncores)
except ImportError:
    ne = None
    print('numexpr could not be imported, only single-thread calculations will '
          'be used. Recommend abort and install numexpr!')


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


def slice_for_step(whole_arr_slice, step, axis):
    """
    Returns the array slice for a given qns accumulation step, given the
    general slice for generating the whole qns array (See general slices
    in hoggsumexp2). The accumulation steps (qns_step in hoggsumexp2 below)
    only have 3 axes, but the slice needs a 4th index in the case that a
    specific row/col/etc is needed along the accumulation axis. E.g. suppose
    axis=2, the following input slices on the left produce the outputs on the
    right:
    (:, :, :, None) --> (:, :, step, None)
    (:, :, None, :) --> (:, :, :)
    Note that both of the above produce a 3D array when used as an indexing
    slice, e.g. lnzs[:, :, step, None] is 3D as is lnzs[:, :, :]

    Note this *could* use a tuple comprehension instead, but comprehensions
    can't be jit compiled by numba.
    """
    tup = tuple()
    for ax_num, ax_slice in enumerate(whole_arr_slice):
        if ax_slice is not None or ax_num != axis:
            if ax_num == axis and ax_slice is not None:
                tup = tup + (step, )
            else:
                tup = tup + (ax_slice, )
    return tup


def hoggsumexp2(lnas, lnzs, axis=None):
    """
    # porpose: 
    - Computes the real function: 'L = log(sum(sum(exp(lnas + lnzs + lnas + conj(lnzs)))))'
    - Computes the gradient of L with respect to the lnas

    # input
    - 'lnas': ndarray of shape (n1, n2, n3, ..., nD, N) of real numbers
    - 'lnzs': ndarray of shape (n1, n2, n3, ..., nD, N) of complex numbers
    - 'axis': the last axis on which to do the sum. sum is done on two dimensions!

    # output:
    - 'L': ndarray of shape (n1, n2, n3, ..., nD) of real numbers
    - 'dL_dlnas': ndarray of shape (n1, n2, n3, ..., nD, N) of real numbers
    """
    if axis is None:
        axis = len(lnzs.shape) # I expand the dimension of lnzs for the next calculation, so I need an extra dimension here!

    # NEW METHOD HERE

    # Shape of qns is same as lnzs with duplicated spatial axis. expLs is same
    # as qns with summation axis gone.
    qns_shape = lnzs.shape + (lnzs.shape[-1], )
    expLs_shape = qns_shape[:axis] + qns_shape[axis + 1:]
    # General slices for the whole qns/dqn arrays
    # Note slice(None) is the same as : in slice notation, e.g. slice_lnzs_1
    # below produces the same slice as lnzs[:, :, :, None]
    slice_lnas_1 = (None, None, None, slice(None))
    slice_lnas_2 = (None, None, slice(None), None)
    slice_lnzs_1 = (slice(None), slice(None), slice(None), None)
    slice_lnzs_2 = (slice(None), slice(None), None, slice(None))

    # Pre-traverse just to calculate Qs for the whole array.
    # Gotta be a better way, but let's make it work first.
    Q = -np.inf
    DQ = -np.inf
    expLs = np.zeros(expLs_shape, dtype=lnzs.dtype)
    expD = np.zeros_like(expLs)
    # Execute 2 iteration passes. First finds Q and DQ max. Second accumulates.
    for iter_pass in ('Q_DQ', 'accum'):
        for step in range(qns_shape[axis]):
            slice_lnas_1_step = slice_for_step(slice_lnas_1, step, axis)
            slice_lnas_2_step = slice_for_step(slice_lnas_2, step, axis)
            slice_lnzs_1_step = slice_for_step(slice_lnzs_1, step, axis)
            slice_lnzs_2_step = slice_for_step(slice_lnzs_2, step, axis)
            # Only perform these slices once and re-use sliced arrays
            lnas_1_step = lnas[slice_lnas_1_step]
            lnas_2_step = lnas[slice_lnas_2_step]
            lnzs_1_step = lnzs[slice_lnzs_1_step]
            lnzs_2_step = lnzs[slice_lnzs_2_step]

            if ne is not None:
                qn1_step = ne.evaluate('lnzs_1_step + conj(lnzs_2_step)')
                qn2_step = ne.evaluate('lnzs_2_step + conj(lnzs_1_step)')
                qns_step = ne.evaluate('lnas_1_step + lnas_2_step + qn1_step')
            else:
                qn1_step = lnzs_1_step + np.conj(lnzs_2_step)
                qn2_step = lnzs_2_step + np.conj(lnzs_1_step)
                qns_step = lnas_1_step + lnas_2_step + qn1_step

            ZQ = np.max((np.max(qn1_step), np.max(qn2_step)))

            if ne is not None:
                dqn_step = ne.evaluate('lnas_1_step + log(exp(qn1_step - ZQ) + exp(qn2_step - ZQ)) + ZQ')
            else:
                dqn_step = lnas_1_step + np.log(np.exp(qn1_step - ZQ) + np.exp(qn2_step - ZQ)) + ZQ

            # First pass will stop here, only finding max of all qns and all dqns
            if iter_pass == 'Q_DQ':
                Q = np.max((Q, np.max(qns_step)))
                DQ = np.max((DQ, np.max(dqn_step)))
                continue

            if ne is not None:
                expqns_step = ne.evaluate('exp(qns_step - Q)')
                expDs_step = ne.evaluate('exp(dqn_step - DQ)')
            else:
                expqns_step = np.exp(qns_step - Q)
                expDs_step = np.exp(dqn_step - DQ)

            expLs += expqns_step
            expD += expDs_step

    if ne is not None:
        # Some bug when supplying axis as a variable in numexpr, so just
        # use format to insert the literal value in the string
        expL = ne.evaluate('sum(expLs, {:d})'.format(axis-1))
        L = ne.evaluate('real(log(expL) + Q)')
        expanded_expL = np.expand_dims(expL, axis-1)
        dL_dlnas = ne.evaluate('real(expD / expanded_expL)')

        return L, dL_dlnas
    else:
        expL = np.sum(expLs, axis=axis-1)  # Perform second sum normally
        L = np.log(expL) + Q
        dL_dlnas = expD / np.expand_dims(expL, axis-1)

        return np.real(L), np.real(dL_dlnas)

    # END NEW METHOD

    # qns = lnas[None, None, None, :] + lnas[None, None, :, None] + lnzs[:, :, :, None] + np.conj(lnzs[:, :, None, :])
    # Q = np.max(qns)
    # expqns = np.exp(qns - Q)
    # expLs = np.sum(expqns, axis=axis)
    # expL = np.sum(expLs, axis=axis-1) # two sums!
    # L = np.log(expL) + Q
    #
    # # calculate the things I need for the derivative:
    # qn1 = lnzs[:, :, :, None] + np.conj(lnzs[:, :, None, :])
    # qn2 = lnzs[:, :, None, :] + np.conj(lnzs[:, :, :, None])
    # Q1 = np.max(qn1)
    # Q2 = np.max(qn2)
    # Q = np.max((Q1, Q2)) # I'm not sure that this is the correct way to go, need to make sure with Hogg!
    #
    # dqn = lnas[None, None, None, :] + np.log(np.exp(qn1 - Q) + np.exp(qn2 - Q)) + Q
    # DQ = np.max(dqn)
    # expDs = np.exp(dqn - DQ)
    # expD = np.sum(expDs, axis=axis)
    #
    # dL_dlnas = expD / np.expand_dims(expL, axis-1)
    #
    # assert np.isclose(L, np.conj(L)).all()
    # assert np.isclose(dL_dlnas, np.conj(dL_dlnas)).all()
    #
    # return np.real(L), np.real(dL_dlnas) # If I passed the assert, I can ignore the complex numbers

class ImageModel:

    def __init__(self, ns, knqs):
        self.N = int(np.max(ns)) + 1
        self.ns = ns
        self.knqs = knqs
        self.initialize_bases()
        self.create_angle_sampling()
        print("image_model:", self.lnams.shape, self.ns.shape, self.xms.shape,
              self.knqs.shape)

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
        self.xms = np.vstack((yms, xms, zms)).T # These are the locations of the Gaussians in REAL space!
        self.lnams = np.random.normal(size=yms.shape)
        return None

    def create_angle_sampling(self, T=2**5):  # MAGIC 1024
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

    def evaluate_rotated_lnbases_cmplx(self, kqs):
        """
        # input:
        - kqs: ndarray of shape [Q, 2]

        # output:
        - lngtqms: evaluations of shape [T, Q, M], the output is complex!
        """
        Q, two = kqs.shape
        assert two == 2
        
        xtms = np.sum(self.rotations[:, None, :, :] * self.xms[None, :, None, :], axis=3) # I am rotating the Gaussians in real space!
        print(kqs.shape, xtms.shape)

        lngtqm = -np.log(np.sqrt(2. * np.pi * self.sigma2)) - 1j * np.sum((kqs[None, :, None, :] * xtms[:, None, :, :]) - self.sigma2/2.0 * kqs[None, :, None, :]**2, axis=3)
        return lngtqm

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
        kqs = (self.knqs[I]).reshape((Q, 2))
        lngtqm = self.evaluate_rotated_lnbases_cmplx(kqs) # we are rotating the photons, lngtqm is complex!
        assert lngtqm.shape == (self.T, Q, self.M)

        # logsumexp over m index and m' index (ie, summing the mixture of Gaussians)
        lnLntqs, dlnLntqs_dlnams = hoggsumexp2(self.lnams, lngtqm, axis=3)
        assert lnLntqs.shape == (self.T, Q)
        assert dlnLntqs_dlnams.shape == (self.T, Q, self.M)
        print("I DID ITTTTTT")

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

    photons = np.load('./photons_None_fft.npy')
    ## ns = photons[:, 0] ## for real space
    ## xnqs = photons[:, 1:] ## for real space
    ns = photons[:, 0]
    knqs = photons[:, 1:]

    # initialize model
    ## model = ImageModel(ns, xnqs) ## for real space
    model = ImageModel(ns, knqs)


    # take a few gradient steps
    fig = plt.figure()
    sumh = 0.
    hplot = 0.
    for j in range(2 ** 16):
        h = 4096. / (2048. + float(j)) # magic
        sumh += h
        n = np.random.randint(model.N) # chosing a random image to work with in the stochastic grad.
        Ln, gradLn = model.single_image_lnlike(n) # calculating the gradient for a single image
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
