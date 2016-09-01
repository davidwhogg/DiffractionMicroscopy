import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mplimg


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
    insert_locations = cdf_indexes[insert_locations]
    indexes_3d = np.unravel_index(insert_locations, image.shape)
    indexes_3d = np.column_stack(indexes_3d)
    jitter = np.random.uniform(size=indexes_3d.size).reshape(indexes_3d.shape)
    return indexes_3d + jitter - np.array(image.shape) / 2


def project_by_random_matrix(photon_zyxs, distort='quadrupole', return_matrix=False):
    """
    Generate a randomized 3D-to-2D projection matrix, and project given photon
    positions using it.

    :param photon_zyxs: Photon positions in 3D, zyx order
    :param distort: Either None, dipole, or quadrapole

    :return: Projected photon positions in 2D, yx order
    """
    # TODO: Two randomly drawn vectors *might* almost dot to 1. Better to just
    # select one axis and then a uniform angle?
    rand_tan1 = np.random.normal(size=3)
    if distort == 'dipole':
        rand_tan1 += (0.9, 0, 0)
    elif distort == 'quadrupole':
        rand_tan1 *= (1.9, 1.0, 1.0)
    rand_tan1 = rand_tan1 / np.sqrt(np.dot(rand_tan1, rand_tan1))
    rand_tan2 = np.random.normal(size=3)
    rand_tan2 -= np.dot(rand_tan1, rand_tan2) * rand_tan1
    rand_tan2 /= np.sqrt(np.dot(rand_tan2, rand_tan2))
    assert np.isclose(np.dot(rand_tan1, rand_tan2), 0.0)

    projection_matrix = np.vstack((rand_tan1, rand_tan2))
    if return_matrix:
        return projection_matrix

    projected_yxs = np.dot(projection_matrix, photon_zyxs.T).T

    return projected_yxs


def make_fake_data(truth, N=1024, rate=1., distort=None):
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
    cdf_indexes = cdf_indexes.reshape(truth.shape)
    cdf_indexes = cdf_indexes[cdf_mask]

    photon_zyxs = get_photon_positions(truth, cdf, cdf_indexes,
                                       nphot=np.sum(Qs))

    row = 0
    for n, Q in enumerate(Qs):
        photon_zyxs_img = photon_zyxs[row:row+Q, :]
        photon_yxs = project_by_random_matrix(photon_zyxs_img, distort=distort)
        nyxs[row:row+Q, :] = np.column_stack([np.repeat(n, Q), photon_yxs])
        row += Q

        # Print progress
        next_pct = 100 * (n + 1) // N
        curr_pct = 100 * n // N
        if next_pct - curr_pct > 0:
            print('{:d}%'.format(next_pct))
    return nyxs[:, 0], nyxs[:, 1], nyxs[:, 2]


def make_truth(img_file='./truth.png'):
    """
    Load in the truth image data, embed it into a 3d array, then rotate it in a
    weird way
    """
    pixels = 1.0 - mplimg.imread(img_file)[:, :, 0]  # b&w image, just grab red

    # Now embed in 3d array and rotate in some interesting way
    voxels = np.zeros(pixels.shape + (pixels.shape[0], ))
    voxels[:, :, voxels.shape[2] // 2] = pixels
    rot_ang_axis = np.array((2, 1, 0.5))  # Something "interesting"
    aff_matrix = angle_axis_to_matrix(rot_ang_axis)
    # Rotate about center, but affine_transform offset parameter is dumb
    center = np.array(voxels.shape) / 2  # whatever close enough
    offset = -(center - center.dot(aff_matrix)).dot(np.linalg.inv(aff_matrix))
    voxels = ndimage.affine_transform(voxels, aff_matrix, offset=offset)

    # Remake the truth figure in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = np.meshgrid(np.arange(voxels.shape[0]),
                          np.arange(voxels.shape[1]),
                          np.arange(voxels.shape[2]),
                          indexing='ij')
    disp_vox = voxels > 0.3
    ax.scatter(x[disp_vox], y[disp_vox], z[disp_vox])
    plt.savefig(img_file.replace('.png', '_3d.png'))
    # plt.show()

    print("truth:", voxels.shape)
    return voxels


def angle_axis_to_matrix(angle_axis):
    """
    Generate a rotation matrix given a rotation represented in angle-axis form

    :param angle_axis: Rotation represented in angle-axis form (vector direction
    is angle, vector length is angle in radians)
    :return: Same rotation in matrix form
    """
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


def test_project_by_random_matrix(nsamples=2**11):
    for distort in (None, 'dipole', 'quadrupole'):
        # We'll just plot the first random projection vector rather than the
        # second orthogonalized one.
        proj_tans = np.zeros((nsamples, 3), dtype=float)
        for sample in range(nsamples):
            matrix = project_by_random_matrix(proj_tans[sample],
                                              distort=distort,
                                              return_matrix=True)
            proj_tans[sample] = matrix[0]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(proj_tans[:, 2], proj_tans[:, 1], proj_tans[:, 0])
        plt.show()


if __name__ == '__main__':
    # test_project_by_random_matrix()

    # Repeatability
    np.random.seed(42)

    truth = make_truth('./truth.png')

    for distort in (None, 'dipole', 'quadrupole'):
        # make fake data
        ns, ys, xs = make_fake_data(truth, N=2**14, rate=2**4, distort=distort)
        xnqs = np.vstack((ys, xs)).T

        print("fake data:", ns.shape, xnqs.shape)

        # Pickle photon location info and random number generator state
        np.save('./photons_{}.npy'.format(distort), np.column_stack([ns, xnqs]))

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
            pfn = "./datum_{:06d}_{}.png".format(n, distort)
            plt.savefig(pfn)
            print(pfn)
