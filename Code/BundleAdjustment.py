import numpy as np
from scipy.sparse import lil_matrix
import scipy.sparse
import scipy.optimize
import scipy.spatial.transform.rotation
import time
from scipy.optimize import least_squares
import BuildVisibilityMatrix as bvm


def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = (n_cameras+1) * 9 + n_points * 3
    A = lil_matrix((m, n), dtype=int)
    i = np.arange(camera_indices.size)
    for s in range(9):
        A[2 * i, camera_indices * 9 + s] = 1
        A[2 * i + 1, camera_indices * 9 + s] = 1

    for s in range(3):

        A[2 * i, n_cameras * 9 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 9 + point_indices * 3 + s] = 1


    return A


def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    numcam = n_cameras+1
    camera_params = params[:numcam * 9].reshape((numcam, 9))
    points_3d = params[numcam * 9:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices])
    return (points_proj - points_2d).ravel()

def project(points, camera_params):
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


def rotate(points, rot_vecs):
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v

def BundleAdjustment(pts_3d,pts_3D_flag, feature_x, feature_y, filtered_feature_flag, Rs, Cs, K, num_cams):
    vis_mat, rec_idxs = bvm.get_visibility_matrix(pts_3D_flag, filtered_feature_flag, num_cams)
    rec_pts_2d = bvm.get_img_coords(rec_idxs, vis_mat, feature_x, feature_y)
    rec_pts_3d = pts_3d[rec_idxs]
    cam_idxs, pts_idxs = bvm.get_cam_idxs(vis_mat)
    num_rec_pts = len(rec_pts_3d)
    cam_params = []
    for i in range(num_cams+1):
        R_, C_ = Rs[i], Cs[i]
        quat_rvec = scipy.spatial.transform.Rotation.from_matrix(R_).as_rotvec()
        cam_params.append([quat_rvec[0], quat_rvec[1], quat_rvec[2], C_[0], C_[1], C_[2], K[0][0], 0, 0])
    cam_params = np.array(cam_params)
    cam_params = np.reshape(cam_params, (-1, 9))
    A = bundle_adjustment_sparsity(num_cams, rec_idxs.shape[0], cam_idxs, pts_idxs)
    x0 = np.hstack((cam_params.ravel(), rec_pts_3d.ravel()))

    t0 = time.time()
    res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-10, method='trf',
                        args=(num_cams, rec_pts_3d.shape[0], cam_idxs, pts_idxs, rec_pts_2d))
    t1 = time.time()
    print('time to run BA :', t1-t0, 's \nA matrix shape: ' ,  A.shape, '\n############')

    params = res.x
    number_of_cam = num_cams + 1
    opt_cam_params = params[:number_of_cam * 9].reshape((number_of_cam, 9))
    opt_pts_3d = params[number_of_cam * 9:].reshape((rec_pts_3d.shape[0], 3))

    opt_pts_3d_ = np.zeros_like(pts_3d)
    opt_pts_3d_[rec_idxs] = opt_pts_3d

    for i in range(num_cams):
        quat_rvec[0] = opt_cam_params[i, 0]
        quat_rvec[1] = opt_cam_params[i, 1]
        quat_rvec[2] = opt_cam_params[i, 2]
        Rs[i] = scipy.spatial.transform.Rotation.from_rotvec([quat_rvec[0], quat_rvec[1], quat_rvec[2]]).as_matrix()
        Cs[i] = np.array([opt_cam_params[i, 3], opt_cam_params[i, 4], opt_cam_params[i, 5]])

    return Rs, Cs, opt_pts_3d_


