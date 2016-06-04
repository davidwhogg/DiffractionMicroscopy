"""
This file is part of the DiffractionMicroscopy project.
Copyright 2016 David W. Hogg (NYU, SCDA).

This piece of code does nothing related to diffraction.
It only shows that you can reconstruct an image from small numbers of
photons taken in exoposures at unknown orientations.
"""
import numpy as np

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
    ns = []
    qs = []
    ys = []
    xs = []
    for n in range(N):
        Q = 0
        while(Q == 0):
            Q = np.random.poisson(rate)
        tys, txs = make_fake_image(truth, Q)
        for q in range(Q):
            ns.append(n)
            qs.append(q)
            ys.append(tys[q])
            xs.append(txs[q])
    return np.array(ns), np.array(qs), np.array(ys), np.array(xs)

if __name__ == "__main__":
    import pylab as plt
    np.random.seed(42)

    truth = make_truth()

    ns, qs, ys, xs = make_fake_data(truth)
    print(ns.shape, qs.shape, ys.shape, xs.shape)

    plt.figure()
    plt.clf()
    plt.plot(xs, ys, 'k.', alpha=0.5)
    plt.axis("equal")
    plt.savefig("./test.png")
