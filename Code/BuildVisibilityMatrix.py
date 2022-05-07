import numpy as np


def get_visibility_matrix(X_found, valid_feature_idxs, num_cams):
    recon_bin = np.array([0]*len(valid_feature_idxs))
    for i in range(num_cams+1):
        recon_bin = recon_bin | valid_feature_idxs[:,i]
    rec_idxs = np.where((X_found.reshape(-1)) & (recon_bin))
    vis_mat = X_found[rec_idxs].reshape(-1,1)
    for n in range(num_cams+1):
        vis_mat = np.hstack((vis_mat, valid_feature_idxs[rec_idxs, n].reshape(-1, 1)))
    return vis_mat[:, 1:-1], rec_idxs[0]


def get_img_coords(rec_idxs, vis_mat, feat1, feat2):
    pts_2d = []
    vis_feat1, vis_feat2 = feat1[rec_idxs], feat2[rec_idxs]
    for i in range(vis_mat.shape[0]):
        for j in range(vis_mat.shape[1]):
            if vis_mat[i,j]:
                pts_2d.append([vis_feat1[i,j], vis_feat2[i,j]])
    return np.array(pts_2d)


def get_cam_idxs(vis_mat):
    cam_idxs = []
    pts_idxs = []
    for i in range(vis_mat.shape[0]):
        for j in range(vis_mat.shape[1]):
            if vis_mat[i,j]:
                cam_idxs.append(j)
                pts_idxs.append(i)
    return np.array(cam_idxs), np.array(pts_idxs)
