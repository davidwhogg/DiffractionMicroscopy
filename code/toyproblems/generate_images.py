import numpy as np
import matplotlib.pyplot as plt
import png
from scipy import ndimage
from mpl_toolkits.mplot3d import Axes3D


def get_photon_positions(image, cdf, cdf_indexes, nphot=1):
    """
    Uses an inverse CDF lookup to find positions for uniform draws

    :param image: The 3d voxel representation of the truth
    :param cdf: CDF representation of the image. CDF should only be computed for
        non-zero pixels
    :param cdf_indexes: 1d indexes from image of the pixels represented in cdf
    :param nphot: Number of photons to draw

    :return: 3D positions of the drawn photons, about the image center

    ISSUES: make sure the cdf picker is statistically correct
    """
    draws = np.random.uniform(size=nphot) * cdf[-1]

    insert_locations = np.searchsorted(cdf, draws)
    argmin_location = np.argmin(np.abs(cdf - draws[0]))
    # assert insert_locations[0] == argmin_location
    insert_locations = cdf_indexes[insert_locations]
    indexes_3d = np.unravel_index(insert_locations, image.shape)
    indexes_3d = np.column_stack(indexes_3d)
    jitter = np.random.uniform(size=indexes_3d.size).reshape(indexes_3d.shape)
    return indexes_3d + jitter - np.array(image.shape) / 2


def project_by_random_matrix(photon_zyxs):
    """
    Generate a randomized 3D-to-2D projection matrix, and project given photon
    positions using it.

    :param photon_zyxs: Photon positions in 3D, zyx order

    :return: Projected photon positions in 2D, yx order
    """
    # TODO: Two randomly drawn vectors *might* almost dot to 1. Better to just
    # select one axis and then a uniform angle?
    rand_tan1 = np.random.normal(size=3)
    rand_tan1 = rand_tan1 / np.sqrt(np.dot(rand_tan1, rand_tan1))
    rand_tan2 = np.random.normal(size=3)
    rand_tan2 -= np.dot(rand_tan1, rand_tan2) * rand_tan1
    rand_tan2 /= np.sqrt(np.dot(rand_tan2, rand_tan2))
    assert np.isclose(np.dot(rand_tan1, rand_tan2), 0.0)

    projection_matrix = np.vstack((rand_tan1, rand_tan2))

    projected_yxs = np.dot(projection_matrix, photon_zyxs.T).T

    return projected_yxs


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
    Qs = np.zeros(N, dtype=int)
    while True:
        Qs[Qs == 0] = np.random.poisson(rate, size=np.sum(Qs == 0))
        if np.all(Qs > 0):
            break

    nyxs = np.zeros(shape=(np.sum(Qs), 3))

    # Only calculate image CDF for pixels with non-zero values.
    # Need to preserve their indexes also.
    cdf_mask = truth > 0
    cdf = np.cumsum(truth[cdf_mask])
    cdf_indexes = np.arange(truth.size, dtype=int)
    cdf_indexes = cdf_indexes.reshape((truth.shape))
    cdf_indexes = cdf_indexes[cdf_mask]

    photon_zyxs = get_photon_positions(truth, cdf, cdf_indexes,
                                       nphot=np.sum(Qs))

    row = 0
    for n, Q in enumerate(Qs):
        photon_zyxs_img = photon_zyxs[row:row+Q, :]
        photon_yxs = project_by_random_matrix(photon_zyxs_img)
        nyxs[row:row+Q, :] = np.column_stack([np.repeat(n, Q), photon_yxs])
        row += Q

        # Print progress
        next_pct = 100 * (n + 1) // N
        curr_pct = 100 * n // N
        if next_pct - curr_pct > 0:
            print('{:d}%'.format(next_pct))
    return nyxs[:,0], nyxs[:,1], nyxs[:,2]


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
    voxels[:, :, 40] = pixels
    rot_ang_axis = np.array((2, 1, 0.5))  # Something "interesting"
    # rot_ang_axis = np.array((0, 1.4, 0))
    aff_matrix = angle_axis_to_matrix(rot_ang_axis)
    center = np.array(voxels.shape)/2  # whatever close enough
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
    s_ang = np.sin(ang)
    c_ang = np.cos(ang)

    out_matrix = c_ang * np.eye(3)

    out_matrix += s_ang * np.array(((0, -axis[2], axis[1]),
                                    (axis[2], 0, -axis[0]),
                                    (-axis[1], axis[0], 0)))

    out_matrix += (1 - c_ang) * np.outer(axis, axis)

    return out_matrix


if __name__ == '__main__':
    # Repeatability
    np.random.seed(42)

    # make fake data
    truth = make_truth()
    ns, ys, xs = make_fake_data(truth, N=2**10, rate=2**8)
    xnqs = np.vstack((ys, xs)).T

    print("fake data:", ns.shape, xnqs.shape)

    # plot first 20 fake data
    plt.figure()
    for n in range(20):
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
    np.save('./photons.npy', (ns, xnqs))
