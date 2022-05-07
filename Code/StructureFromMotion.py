import Utils as ut
import matplotlib.pyplot as plt

import cv2
import numpy as np
import PnPRansac as lpp
import NonLinearPnP as nlp
import LinearTriangulation as lt
import NonLinearTriangulation as nlt
import BundleAdjustment as ba
import scipy.spatial.transform.rotation


def run(input_dir, output_dir, num_images):

    # read all images
    images = ut.read_images(input_dir, num_images)

    # extract features from the matching text files
    x_features, y_features = ut.get_matching_features(input_dir, num_images)

    # f_matrix_list, inliers_list = ut.get_fundamental_matrices(x_features, y_features, num_images)
    f_matrices, inlier_idx_flag = ut.get_inliers(x_features, y_features, num_images, images, output_dir)

    # register 1st and 2nd image
    first = 0
    second = 1
    K = ut.read_calibration()

    f_mat_1_2 = f_matrices[first, second]
    e_mat_1_2 = ut.get_essential_matrices([f_mat_1_2], K)
    r_list, c_list = ut.get_camera_poses(e_mat_1_2)

    inlier_idx_flag = np.int32(inlier_idx_flag)
    idx = np.where(inlier_idx_flag[:, first] & inlier_idx_flag[:, second])
    img1_pts = np.hstack((x_features[idx, first].reshape(-1, 1), y_features[idx, first].reshape(-1, 1)))
    img2_pts = np.hstack((x_features[idx, second].reshape(-1, 1), y_features[idx, second].reshape(-1, 1)))

    print('[INFO]: Registering Images 1 and 2')
    # only for first two images
    pts_3D_list = ut.get_pts_3D(K, r_list, c_list, [[img1_pts, img2_pts]], linear=True, pts_3D_list=None)

    # only for first two images
    best_r_list, best_c_list, best_pts_3D_list = ut.get_best_R_and_C_and_pts_3D(pts_3D_list, r_list, c_list)
    pts_3D = best_pts_3D_list[0]
    mean_error_lt = ut.get_mean_reprojection_error(K=K, pts_3D=pts_3D, pts1=img1_pts, pts2=img2_pts, R1=None, C1=None,
                                                   R2=best_r_list[0], C2=best_c_list[0])

    print("[INFO]: Mean Reprojection Error after Linear Triangulation ", mean_error_lt)

    pts_3d_list_refined = ut.get_pts_3D(K, best_r_list, best_c_list, [[img1_pts, img2_pts]],
                                        linear=False, pts_3D_list=best_pts_3D_list)

    X_refined = pts_3d_list_refined[0].reshape(-1, 4)
    X_refined = X_refined / X_refined[:, 3].reshape(-1, 1)

    mean_error_nlt = ut.get_mean_reprojection_error(K=K, pts_3D=X_refined, pts1=img1_pts, pts2=img2_pts,
                                                    R1=None, C1=None, R2=best_r_list[0], C2=best_c_list[0])

    print("[INFO]: Mean Reprojection Error after Non-Linear Triangulation ", mean_error_nlt)

    pts_3D_all = np.zeros((x_features.shape[0], 3))
    pts_3D_flag = np.zeros((x_features.shape[0], 1), dtype=int)
    pts_3D_all[idx] = pts_3D[:, :3]
    pts_3D_flag[idx] = 1
    pts_3D_flag[np.where(pts_3D_all[:, 2] < 0)] = 0

    C_set_ = []
    R_set_ = []

    C0 = np.zeros(3)
    R0 = np.identity(3)
    C_set_.append(C0)
    R_set_.append(R0)

    C_set_.append(best_c_list[0])
    R_set_.append(best_r_list[0])

    print('[INFO]: Registered Image 1 and 2')

    for i in range(2, num_images):
        print('[INFO]: Registering Image: {0}. Please Wait for Some time'.format(str(i + 1)))

        feature_idx_i = np.where(pts_3D_flag[:, 0] & inlier_idx_flag[:, i])
        if len(feature_idx_i[0]) < 8:
            print("[WARN]: Cannot Register Image {0}, Number of Correspondences {1}. SKIPPING image {2}".format(i + 1, len(feature_idx_i[0]), i + 1))
            continue

        pts_2D = np.hstack((x_features[feature_idx_i, i].reshape(-1, 1), y_features[feature_idx_i, i].reshape(-1, 1)))
        pts_3D = pts_3D_all[feature_idx_i, :].reshape(-1, 3)
        R_init, C_init = lpp.PnPRANSAC(K, pts_2D, pts_3D)
        mean_reproj_error_lpnp = lpp.reprojection_errorPnP(pts_3D, pts_2D, K, R_init, C_init)

        Ri, Ci = nlp.NonLinearPnP(pts_3D, pts_2D, K, R_init, C_init)
        mean_reproj_error_nlpnp = lpp.reprojection_errorPnP(pts_3D, pts_2D, K, Ri, Ci)
        print("[INFO]: Mean Reprojection Error after Linear PnP: ", mean_reproj_error_lpnp)
        print("[INFO]: Mean Reprojection Error after Non-Linear PnP: ", mean_reproj_error_nlpnp)

        C_set_.append(Ci)
        R_set_.append(Ri)

        for j in range(0, i):
            idx_X_pts = np.where(inlier_idx_flag[:, j] & inlier_idx_flag[:, i])
            if len(idx_X_pts[0]) < 8:
                print("[WARN]: Cannot Add new 3D points of Image {0}, Number of Correspondences {1}."
                      " SKIPPING image {2}".format(j + 1, len(idx_X_pts[0]), i + 1))
                continue

            x1 = np.hstack((x_features[idx_X_pts, j].reshape((-1, 1)), y_features[idx_X_pts, j].reshape((-1, 1))))
            x2 = np.hstack((x_features[idx_X_pts, i].reshape((-1, 1)), y_features[idx_X_pts, i].reshape((-1, 1))))

            I = np.identity(3)
            C1 = C_set_[j].reshape(-1, 1)
            P1 = np.dot(K, np.dot(R_set_[j], np.hstack((I, -C1))))

            C2 = Ci.reshape(-1, 1)
            P2 = np.dot(K, np.dot(Ri, np.hstack((I, -C2))))
            pts_3D= lt.linear_triangulation(P1, P2, x1, x2)
            pts_3D = pts_3D/pts_3D[:,3].reshape(-1,1)

            mean_error_lt = ut.get_mean_reprojection_error(K=K, pts_3D=pts_3D, pts1=x1, pts2=x2, R1=R_set_[j],
                                                           C1=C_set_[j], R2=Ri, C2=C2)
            print("[INFO]: Mean Reprojection Error after Linear Triangulation ", mean_error_lt)

            pts_3D = nlt.non_linear_triangulation(x1, x2, pts_3D, P1, P2)
            pts_3D = pts_3D/pts_3D[:,3].reshape(-1,1)
            mean_error_nlt = ut.get_mean_reprojection_error(K=K, pts_3D=pts_3D, pts1=x1, pts2=x2, R1=R_set_[j],
                                                           C1=C_set_[j], R2=Ri, C2=C2)
            print("[INFO]: Mean Reprojection Error after Non-Linear Triangulation ", mean_error_nlt)

            pts_3D_all[idx_X_pts] = pts_3D[:, :3]
            pts_3D_flag[idx_X_pts] = 1

            print('[INFO]: Started Bundle Adjustment  for image {} '.format(i+1))
            R_set_, C_set_, pts_3D_all = ba.BundleAdjustment(pts_3D_all, pts_3D_flag, x_features, y_features,
                                                     inlier_idx_flag, R_set_, C_set_, K, i)

            for k in range(0, i + 1):
                idx_X_pts = np.where(pts_3D_flag[:, 0] & inlier_idx_flag[:, k])
                x = np.hstack((x_features[idx_X_pts, k].reshape((-1, 1)), y_features[idx_X_pts, k].reshape((-1, 1))))
                pts_3D = pts_3D_all[idx_X_pts]
                BA_error = lpp.reprojection_errorPnP(pts_3D, x, K, R_set_[k], C_set_[k])
                print("[INFO]: Mean Reprojection Error after Bundle Adjustment {}".format(BA_error))

    pts_3D_flag[pts_3D_all[:, 2] < 0] = 0
    ut.draw_plots(pts_3D_all, pts_3D_flag, R_set_, C_set_)


if __name__ == "__main__":
    input_dir = "../Data/input"
    output_dir = "../Data/output"
    num_images = 6

    run(input_dir, output_dir, num_images)
