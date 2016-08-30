import numpy as np
import matplotlib.pyplot as plt
import png


def get_one_photon(image):
    """
    stupidly uses rejection sampling!
    lots of annoying details.
    """
    h, w = image.shape
    maxi = np.max(image)
    count = 0
    while count == 0:
        yy = np.random.randint(h)
        xx = np.random.randint(w)
        if image[yy, xx] > maxi * np.random.uniform():
            count = 1
    return yy - h / 2 + np.random.uniform(), xx - w / 2 + np.random.uniform()


def make_fake_image(truth, Q):
    """
    # inputs:
    - truth: pixelized image of density
    - Q: number of photons to make

    # issues:
    - duplicates the projection operation in the image model class above.
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
    - Images that get zero photons will be dropped, but N images will be
      returned.
    """
    ns, ys, xs = [], [], []
    for n in range(N):
        Q = 0
        while Q == 0:
            Q = np.random.poisson(rate)
        tys, txs = make_fake_image(truth, Q)
        for q in range(Q):
            ns.append(n)
            ys.append(tys[q])
            xs.append(txs[q])
    return np.array(ns), np.array(ys), np.array(xs)


def make_truth():
    """
    OMG dumb.
    dependency: PyPNG
    """
    plt.figure(figsize=(0.5, 0.3))
    plt.clf()
    plt.text(0.5, 0.5, r"Dalya",
             ha="center", va="center",
             clip_on=False,
             transform=plt.gcf().transFigure)
    plt.gca().set_axis_off()
    datafn = "./truth.png"
    plt.savefig(datafn, dpi=200)
    w, h, pixels, metadata = png.Reader(filename=datafn).read_flat()
    pixels = (np.array(pixels).reshape((h, w, 4)))[::-1, :, 0]
    pixels = (np.max(pixels) - pixels) / np.max(pixels)
    print("truth:", w, h, pixels.shape, metadata)
    return pixels


if __name__ == '__main__':
    # Repeatability
    np.random.seed(42)

    # make fake data
    truth = make_truth()
    ns, ys, xs = make_fake_data(truth, N=2 ** 16, rate=4.)
    xnqs = np.vstack((ys, xs)).T

    print("fake data:", ns.shape, xnqs.shape)

    # plot fake data
    plt.figure()
    for n in range(256):
        plt.clf()
        I = (ns == n)
        plt.plot(xs[I], ys[I], 'k.')
        plt.axis("square")
        plt.xlim(-50., 50.)
        plt.ylim(-50., 50.)
        plt.title("image $n={}$ ($Q={}$)".format(n, np.sum(I)))
        pfn = "./datum_{:06d}.png".format(n)
        plt.savefig(pfn)
        print(pfn)

    # Pickle photon location info and random number generator state
    np.save('./photons.npy', (ns, xnqs, np.random.get_state()))
