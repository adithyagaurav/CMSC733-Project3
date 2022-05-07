import numpy as np
import scipy.optimize as opt


def non_linear_triangulation(pts1, pts2, pts_3D, P1, P2):

    refined_pts_3D = list()
    for i in range(len(pts_3D)):
        optimized = opt.least_squares(fun=reprojection_loss, x0=pts_3D[i], method="trf", args=[pts1[i], pts2[i], P1, P2])
        pt_3D = optimized.x
        refined_pts_3D.append(pt_3D)

    return np.array(refined_pts_3D)


def reprojection_loss(X, pt1, pt2, P1, P2):

    p1_1_T = P1[0, :].reshape(1, -1)
    p1_2_T = P1[1, :].reshape(1, -1)
    p1_3_T = P1[2, :].reshape(1, -1)

    p2_1_T = P2[0, :].reshape(1, -1)
    p2_2_T = P2[1, :].reshape(1, -1)
    p2_3_T = P2[2, :].reshape(1, -1)

    pt1_X_proj = np.divide(p1_1_T.dot(X), p1_3_T.dot(X))
    pt1_y_proj = np.divide(p1_2_T.dot(X), p1_3_T.dot(X))
    pt_1_reprojection_error = np.square(pt1[1] - pt1_y_proj) + np.square(pt1[0] - pt1_X_proj)

    pt2_x_proj = np.divide(p2_1_T.dot(X), p2_3_T.dot(X))
    pt2_y_proj = np.divide(p2_2_T.dot(X), p2_3_T.dot(X))
    pt_2_reprojection_error = np.square(pt2[1] - pt2_y_proj) + np.square(pt2[0] - pt2_x_proj)

    total_reprojection_error = pt_1_reprojection_error + pt_2_reprojection_error
    final_loss = total_reprojection_error.squeeze()

    return final_loss