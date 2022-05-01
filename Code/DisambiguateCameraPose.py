import numpy as np


def disambiguate_camera_pose(R, C, pts_3D):
    max_positive_depths = None
    max_positive_depth_idx = None

    for i in range(len(R)):
        R2 = R[i]
        C2 = C[i].reshape(-1, 1)

        r3 = R2[2].reshape(1, -1)
        pts_3D_ = pts_3D[i]
        pts_3D_ /= pts_3D_[3, :]
        pts_3D_ = pts_3D_[:3, :]

        num_depths = get_positive_depth_count(pts_3D_, r3, C2)
        if max_positive_depths is None or num_depths > max_positive_depths:
            max_positive_depths = num_depths
            max_positive_depth_idx = i

    best_R = R[max_positive_depth_idx]
    best_C = C[max_positive_depth_idx]
    best_pts_3D = pts_3D[max_positive_depth_idx]

    return best_R, best_C, best_pts_3D


def get_positive_depth_count(pts, r3, C):
    num_positive_depth = 0
    for i in range(pts.shape[1]):
        pt = pts[:, i]
        pt = pt.reshape(-1, 1)

        if np.matmul(r3, (pt - C)) > 0 and pt[2] > 0:
            num_positive_depth += 1

    return num_positive_depth
