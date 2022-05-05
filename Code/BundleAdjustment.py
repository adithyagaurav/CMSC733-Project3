import numpy as np
import scipy
import scipy.sparse
import scipy.optimize
import scipy.spatial.transform.rotation


def rotate(points, rot_vecs):
    """Rotate points by given rotation vectors.

    Rodrigues' rotation formula is used.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v



def project(points, camera_params):
    """Convert 3-D points to 2-D by projecting onto images."""
    points_proj = rotate(points, camera_params[:, :3])
    points_proj += camera_params[:, 3:6]
    points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    f = camera_params[:, 6]
    k1 = camera_params[:, 7]
    k2 = camera_params[:, 8]
    n = np.sum(points_proj**2, axis=1)
    r = 1 + k1 * n + k2 * n**2
    points_proj *= (r * f)[:, np.newaxis]
    return points_proj




def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    """Compute residuals.

    `params` contains camera parameters and 3-D coordinates.
    """
    camera_params = params[:n_cameras * 9].reshape((n_cameras, 9))
    points_3d = params[n_cameras * 9:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices])
    return (points_proj - points_2d).ravel()

def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 9 + n_points * 3
    A = scipy.sparse.lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(9):
        A[2 * i, camera_indices * 9 + s] = 1
        A[2 * i + 1, camera_indices * 9 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 9 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 9 + point_indices * 3 + s] = 1

    return A

def do_bundle_adjustment(Cs, Rs, pts_3d, K, pts_2d, cam_idxs, vis_mat, reconstruct_pts):
    rec_idxs = np.where(reconstruct_pts==1)[0]
    rec_pts_3d = pts_3d[rec_idxs, :]
    num_rec_pts = len(rec_pts_3d)
    cam_params = []
    num_cams = 0
    for R, C in zip(Rs, Cs):
        num_cams+=1
        f = K[1, 1]
        quat_rvec = scipy.spatial.transform.Rotation.from_dcm(R).as_rotvec()
        cam_params.append([quat_rvec[0], quat_rvec[1], quat_rvec[2], C[0], C[1], C[2], f, 0, 0])

    cam_params = np.array(cam_params)
    cam_params = np.reshape(cam_params, (-1, 9))
    n, m = 9*num_cams + 3*num_rec_pts, 2*len(pts_2d)

    A = bundle_adjustment_sparsity(num_cams, num_rec_pts, cam_idxs, rec_idxs)
    x0 = np.hstack((cam_params.ravel(), pts_3d.ravel()))
    res = scipy.optimize.least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                        args=(num_cams, len(pts_3d), cam_idxs, rec_idxs, rec_idxs))

    parameters = res.x

    camera_p = np.reshape(parameters[0:cam_params.size], (num_cams, 9))

    X = np.reshape(parameters[cam_params.size:], (len(pts_3d), 3))

    for i in range(num_cams):
        quat_rvec[0] = camera_p[i, 0]
        quat_rvec[1] = camera_p[i, 1]
        quat_rvec[2] = camera_p[i, 2]
        C[0] = camera_p[i, 2]
        C[1] = camera_p[i, 2]
        C[2] = camera_p[i, 6]
        Rs[i] = scipy.spatial.transform.Rotation.from_rotvec([quat_rvec[0], quat_rvec[1], quat_rvec[2]]).as_dcm()
        Cs[i] = [C[0], C[1], C[2]]


    return Rs, Cs, pts_3d