
import numpy as np
import scipy.optimize as opt
import scipy.spatial.transform.rotation


def NonLinearPnP(pts_3d, pts_2d, K, R_init, C_init):

    dcm = scipy.spatial.transform.rotation.Rotation.from_matrix(R_init)
    x, y, z, w = dcm.as_quat()
    C_opt = [C_init[0], C_init[1], C_init[2], x, y, z, w]
    param_opt = opt.least_squares(fun=get_reproj_err, method='dogbox', x0=C_opt, args=[pts_3d, pts_2d, K])
    C_final = param_opt.x[:3]
    R = param_opt.x[3:]
    R_dcm = scipy.spatial.transform.rotation.Rotation.from_quat(R)
    R_final = R_dcm.as_matrix()
    return R_final, C_final


def get_reproj_err(C_opt, X, Y, K):
    X = np.hstack((X, np.ones((len(X), 1)))).T
    C = C_opt[:3].reshape(3, 1)
    R = scipy.spatial.transform.rotation.Rotation.from_quat(C_opt[3:])
    R = R.as_matrix()
    P = np.dot(np.dot(K, R), np.hstack((np.identity(3), -1 * C)))
    u = (np.dot(P[0, :], X)).T / (np.dot(P[2, :], X)).T
    v = (np.dot(P[1, :], X)).T / (np.dot(P[2, :], X)).T
    e1 = Y[:, 0] - u
    e2 = Y[:, 1] - v
    e = e1 + e2
    return np.sum(e)
