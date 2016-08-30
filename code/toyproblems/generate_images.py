import numpy as np
import matplotlib.pyplot as plt
import png
from scipy import ndimage
from mpl_toolkits.mplot3d import Axes3D


def get_one_photon(image):
    """
    stupidly uses rejection sampling!
    lots of annoying details.
    """
    d, h, w = image.shape
    maxi = np.max(image)
    count = 0
    while count == 0:
        zz = np.random.randint(d)
        yy = np.random.randint(h)
        xx = np.random.randint(w)
        if image[zz, yy, xx] > maxi * np.random.uniform():
            count = 1
    return (zz - d / 2 + np.random.uniform(),
            yy - h / 2 + np.random.uniform(),
            xx - w / 2 + np.random.uniform())


def make_fake_image(truth, Q):
    """
    # inputs:
    - truth: pixelized image of density
    - Q: number of photons to make

    # issues:
    - duplicates the projection operation in the image model class above.
    """
    rand_tan1 = np.random.normal(size=3)
    rand_tan1 = rand_tan1 / np.sqrt(np.dot(rand_tan1, rand_tan1))
    rand_tan2 = np.random.normal(size=3)
    rand_tan2 -= np.dot(rand_tan1, rand_tan2) * rand_tan1
    rand_tan2 /= np.sqrt(np.dot(rand_tan2, rand_tan2))
    assert np.isclose(np.dot(rand_tan1, rand_tan2), 0.0)

    projection_matrix = np.vstack((rand_tan1, rand_tan2))

    # theta = 2. * np.pi * np.random.uniform()
    # ct, st = np.cos(theta), np.sin(theta)
    ys = np.zeros(Q)
    xs = np.zeros(Q)
    for q in range(Q):
        zyx = get_one_photon(truth)

        xyq = np.dot(projection_matrix, zyx)

        xs[q] = xyq[0]
        ys[q] = xyq[1]
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
    plt.figure(figsize=(0.6, 0.4))
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

    # Now embed in 3d array and rotate in some interesting way
    voxels = np.zeros(pixels.shape+(0.5*200,))
    voxels[:,:,40] = pixels
    rot_ang_axis = np.array((2,1,0.5))
    # rot_ang_axis = np.array((0, 1.4, 0))
    aff_matrix = angle_axis_to_matrix(rot_ang_axis)
    center = np.array(voxels.shape)/2  # whatever
    # affine_transform offset parameter is dumb
    offset = -(center - center.dot(aff_matrix)).dot(np.linalg.inv(aff_matrix))
    voxels = ndimage.affine_transform(voxels, aff_matrix, offset=offset)

    # Remake the truth figure in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x,y,z = np.meshgrid(np.arange(voxels.shape[0]),np.arange(voxels.shape[1]),
                        np.arange(voxels.shape[2]), indexing='ij')
    disp_vox = voxels > 0.3
    ax.scatter(x[disp_vox], y[disp_vox], z[disp_vox])
    plt.savefig(datafn.replace('.png', '_3d.png'))
    # plt.show()

    print("truth:", w, h, voxels.shape, metadata)
    return voxels


def angle_axis_to_matrix(angle_axis):
    ang = np.sqrt(np.dot(angle_axis, angle_axis))
    axis = angle_axis / ang
    cross_matrix = np.array(((0, -axis[2], axis[1]),
                             (axis[2], 0, -axis[0]),
                             (-axis[1], axis[0], 0)))
    return np.cos(ang) * np.eye(3) + np.sin(ang) * cross_matrix + \
           (1 - np.cos(ang)) * np.outer(axis, axis)


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
