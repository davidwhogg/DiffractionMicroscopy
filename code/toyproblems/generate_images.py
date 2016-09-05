import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mplimg
from matplotlib.colors import LogNorm
from numpy import fft


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


def project_by_random_matrix(photon_zyxs, distort=None, debug=False):
    """
    Generate a randomized 3D-to-2D projection matrix, and project given photon
    positions using it.

    :param photon_zyxs: Photon positions in 3D, zyx order
    :param distort: Either None, or a dictionary of vectors {'dipole': vec,
       'quadrupole': vec}
    :param debug: If True, return axis, rot matrix, and proj matrix instead of
        transforming points
    :return: Projected photon positions in 2D, yx order
    """
    rand_axis = np.random.normal(size=3)
    if distort is not None:
        if 'quadrupole' in distort:
            rand_axis *= distort['quadrupole']
        if 'dipole' in distort:
            rand_axis += distort['dipole']
    rand_axis /= np.sqrt(np.dot(rand_axis, rand_axis))
    rand_angle = np.random.uniform(0, 2 * np.pi) + 1  # 0 to 2pi can scale by 0

    rot_matrix = angle_axis_to_matrix(rand_angle*rand_axis)
    proj_matrix = rot_matrix[:, 1:3].T  # first two cols (arbitrary)

    if debug:
        return rand_axis, rot_matrix, proj_matrix

    projected_yxs = np.dot(proj_matrix, photon_zyxs.T).T

    return projected_yxs


def random_fourier_slice(f_image_zyxis, distort=None):
    rand_axis = np.random.normal(size=3)
    if distort is not None:
        if 'quadrupole' in distort:
            rand_axis *= distort['quadrupole']
        if 'dipole' in distort:
            rand_axis += distort['dipole']
    rand_axis /= np.sqrt(np.dot(rand_axis, rand_axis))
    rand_angle = np.random.uniform(0, 2*np.pi) + 1  # 0 to 2pi can scale by 0

    rot_matrix = angle_axis_to_matrix(rand_angle*rand_axis)
    proj_matrix = rot_matrix[:, 1:3].T  # project along z axis (arbitrary)

    # Point-plane distance for a plane in Hessian normal form is just
    # dot(n, x) + p. If the plane goes through the origin p is zero. Get voxels
    # that are within 1 unit of the slicing plane

    z_axis_aug = np.zeros(4)
    z_axis_aug[0:3] = rot_matrix[:, 0]
    slice_plane_distances = np.abs(np.dot(z_axis_aug, f_image_zyxis.T))
    # TODO: be more precise about which voxels to select
    dist_mask = slice_plane_distances < 1.0
    fourier_slice = f_image_zyxis[dist_mask]
    dist_weights = 1 - slice_plane_distances[dist_mask]

    proj_matrix_aug = np.zeros((3, 4))
    proj_matrix_aug[0:2, 0:3] = proj_matrix
    proj_matrix_aug[2, 3] = 1
    proj_yxis = np.dot(proj_matrix_aug, fourier_slice.T).T
    # assert np.all(np.isclose(proj_yxis[:,2], fourier_slice[:,3]))

    # TODO: Right now this sums up into pixels bins. Should really interpolate.
    extents_yx = np.ceil(np.abs(proj_yxis[:, 0:2]).max(axis=0))
    bins_x = np.arange(-extents_yx[1], extents_yx[1]+1)
    bins_y = np.arange(-extents_yx[0], extents_yx[0]+1)

    img2d = np.zeros((bins_x.size-1, bins_y.size-1))
    # 2d gaussian kernel with FWHM=1 pixel
    kern = np.outer((0.05554667,  0.88890666,  0.05554667),
                    (0.05554667,  0.88890666,  0.05554667))
    shifts = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1),
              (1, 0), (1, 1)]
    for (shiftx, shifty), kern_weight in zip(shifts, kern.flat):
        img, bx, by = np.histogram2d(
            proj_yxis[:, 1]+shiftx, proj_yxis[:, 0]+shifty,
            bins=(bins_x, bins_y),
            weights=proj_yxis[:, 2]*dist_weights*kern_weight)
        img2d += img

    return img2d


def make_fake_data_fft(truth, num_images=1024, rate=1., distort=None,
                       save_pngs=0):
    """
    # inputs:
    - truth: pixelized image of density
    - N: number of images to take
    - rate: mean number of photons per image

    # notes:
    - Images that get zero photons will be dropped, but N images will be
      returned.
    """
    n_phots = np.zeros(num_images, dtype=int)
    while True:
        resamp_mask = n_phots == 0
        n_phots[resamp_mask] = np.random.poisson(rate, size=np.sum(resamp_mask))
        if np.all(n_phots > 0):
            break

    nyxs = np.zeros(shape=(np.sum(n_phots), 3))

    fft_truth = fft.fftshift(fft.fftn(truth))
    fft_truth = np.real(fft_truth * fft_truth.conj())
    inds = np.arange(fft_truth.size, dtype=int)
    inds_3d = np.unravel_index(inds, fft_truth.shape)
    fourier_zyxis = np.column_stack(inds_3d + (fft_truth.flatten(),))
    fourier_zyxis -= (fft_truth.shape[2]//2, fft_truth.shape[1]//2,
                      fft_truth.shape[0]//2, 0)

    row = 0
    saved_figs = 0
    if save_pngs > 0:
        plt.figure()
    for n, n_phots_img in enumerate(n_phots):
        img_slice = random_fourier_slice(fourier_zyxis, distort=distort)
        # Only calculate image CDF for pixels with non-zero values.
        # Need to preserve their indexes also.
        cdf_mask = img_slice > 0
        cdf = np.cumsum(img_slice[cdf_mask])
        cdf_indexes = np.arange(img_slice.size, dtype=int)
        cdf_indexes = cdf_indexes.reshape(img_slice.shape)
        cdf_indexes = cdf_indexes[cdf_mask]

        photon_yxs = get_photon_positions(img_slice, cdf, cdf_indexes,
                                          nphot=n_phots_img)

        nyxs_img = np.column_stack([np.repeat(n, n_phots_img), photon_yxs])
        nyxs[row:row+n_phots_img, :] = nyxs_img
        row += n_phots_img

        # Print progress
        next_pct = 100 * (n + 1) // num_images
        curr_pct = 100 * n // num_images
        if next_pct - curr_pct > 0:
            print('{:d}%'.format(next_pct))

        if saved_figs < save_pngs:
            distort_type = ','.join(distort.keys()) if distort else None
            plt.clf()
            plt.imshow(img_slice, cmap='gray_r', norm=LogNorm(), origin='lower',
                       extent=(-img_slice.shape[1]/2, img_slice.shape[1]/2,
                               -img_slice.shape[0]/2, img_slice.shape[0]/2))
            plt.scatter(nyxs_img[:, 2], nyxs_img[:, 1], c='RoyalBlue')
            plt.title("image $n={}$ ($Q={}$)".format(n, n_phots_img))
            img_name = "./datum_{:06d}_{}.png".format(n, distort_type)
            plt.savefig(img_name)
            print(img_name)
            saved_figs += 1
    return nyxs


def make_fake_data(truth, num_images=1024, rate=1., distort=None, save_pngs=0):
    """
    # inputs:
    - truth: pixelized image of density
    - N: number of images to take
    - rate: mean number of photons per image

    # notes:
    - Images that get zero photons will be dropped, but N images will be
      returned.
    """
    Qs = np.zeros(num_images, dtype=int)
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
    saved_figs = 0
    for n, Q in enumerate(Qs):
        photon_zyxs_img = photon_zyxs[row:row+Q, :]
        photon_yxs = project_by_random_matrix(photon_zyxs_img, distort=distort)
        nyxs[row:row+Q, :] = np.column_stack([np.repeat(n, Q), photon_yxs])
        row += Q

        # Print progress
        next_pct = 100 * (n + 1) // num_images
        curr_pct = 100 * n // num_images
        if next_pct - curr_pct > 0:
            print('{:d}%'.format(next_pct))

        if saved_figs < save_pngs:
            distort_type = ','.join(distort.keys()) if distort else None
            plt.clf()
            plt.scatter(photon_yxs[:, 1], photon_yxs[:, 0], c='RoyalBlue')
            plt.title("image $n={}$ ($Q={}$)".format(n, Q))
            img_name = "./datum_{:06d}_{}.png".format(n, distort_type)
            plt.savefig(img_name)
            print(img_name)
            saved_figs += 1
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
    out_matrix += s_ang * np.array(((0, -axis[0], axis[1]),
                                    (axis[0], 0, -axis[2]),
                                    (-axis[1], axis[2], 0)))
    out_matrix += (1 - c_ang) * np.outer(axis[::-1], axis[::-1])
    return out_matrix


def test_project_by_random_matrix(nsamples=2**11):
    """
    Verify that random matrix projection with distortion is working by
    generating many random matrices and ensuring the projection normals are
    distributed as expected
    :param nsamples: Number of sample matrices to generate
    """
    for distort in (None, {'dipole': (1.5, 0.0, 0.0)},
                    {'quadrupole': (3.0, 1.0, 1.0)}):
        # We'll just plot the first random projection vector rather than the
        # second orthogonalized one.
        proj_norms = np.zeros((nsamples, 3), dtype=float)
        for sample in range(nsamples):
            axis, rot_mat, proj_mat = project_by_random_matrix(
                proj_norms[sample], distort=distort, debug=True)
            proj_norms[sample] = axis

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(proj_norms[:, 2], proj_norms[:, 1], proj_norms[:, 0])
        plt.show()


def test_projection_matrix():
    """
    The projection matrix takes a vector in zyx order and projects onto a plane
    with normal defined by the rotation matrix z axis. Output in yx order. This
    test is a visualization of that projection.
    """
    sample_vector = (1.0, 1.0, 1.0)
    axis, rot_mat, proj_mat = project_by_random_matrix(sample_vector,
                                                       debug=True)
    projected = np.dot(proj_mat, sample_vector)

    assert np.isclose(np.dot(rot_mat[:, 0], rot_mat[:, 1]), 0)
    assert np.isclose(np.dot(rot_mat[:, 1], rot_mat[:, 2]), 0)
    assert np.isclose(np.dot(rot_mat[:, 0], rot_mat[:, 2]), 0)

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(121, projection='3d')
    ax.quiver(0, 0, 0, sample_vector[2], sample_vector[1], sample_vector[0],
              pivot='tail', color='black')
    for col, color in [(0, 'blue'), (1, 'green'), (2, 'red')]:
        ax.quiver(0, 0, 0, rot_mat[2, col], rot_mat[1, col], rot_mat[0, col],
                  pivot='tail', color=color)

    ax2d = fig.add_subplot(122)
    ax2d.quiver(0, 0, projected[1], projected[0], pivot='tail', color='black',
                angles='xy', scale_units='xy', scale=1.)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim3d(-1, 1)
    ax.set_title('RGB=xyz. Rotate me so z (blue) \naxis points out of screen')
    ax2d.set_xlim(-1, 1)
    ax2d.set_ylim(-1, 1)
    plt.subplots_adjust(left=0.05, right=0.95)
    plt.show()


def test_slice_theorem():
    """
    TODO: Write a test that verifies the slicing math. I.e it should go the
    other way around the slice theorem -- project 3d to 2d, then do FFT.
    """
    return


if __name__ == '__main__':
    # test_projection_matrix()
    # test_project_by_random_matrix()

    # Repeatability
    np.random.seed(42)

    truth = make_truth('./truth.png')

    for distort in (None, {'dipole': (1.5, 0.0, 0.0)},
                    {'quadrupole': (3.0, 1.0, 1.0)}):
        nyxs = make_fake_data_fft(truth, num_images=2**14, rate=2**4,
                                  distort=distort, save_pngs=20)
        print("fake data:", nyxs.shape)

        # Pickle photon location info
        distort_type = ','.join(distort.keys()) if distort else None
        np.save('./photons_{}.npy'.format(distort_type), nyxs)
