import os
import cv2
import numpy as np
import scipy.optimize

import GetInliersRANSAC as ir
import EssentialMatrixFromFundamentalMatrix as em
import ExtractCameraPose as ep
import LinearTriangulation as lt
import DisambiguateCameraPose as dc
import LinearPnP as lpp

def read_calibration():

    K = np.array([[568.996140852, 0, 643.21055941], [0, 568.988362396, 477.982801038], [0, 0, 1]])

    return K


def read_images(data_dir, num_images):

    images = []
    for i in range(1, num_images + 1):
        img = cv2.imread(os.path.join(data_dir, str(i) + ".jpg"))
        images.append(img)

    return images


def get_matching_features(data_dir, num_images):

    x_features = list()
    y_features = list()

    for i in range(1, num_images):
        match_file_path = os.path.join(data_dir, "matching" + str(i) + ".txt")
        match_file_data = open(match_file_path, "r")
        feature_x = list()
        feature_y = list()

        for idx, line in enumerate(match_file_data):
            if idx != 0:
                row_data = line.split()
                data = list(float(val) for val in row_data)
                data = np.array(data)

                num_matches = data[0]

                src_img_x, src_img_y = data[4], data[5]

                x_row = np.ones((1, num_images)) * (-1)
                y_row = np.ones((1, num_images)) * (-1)

                x_row[0, i-1] = src_img_x
                y_row[0, i-1] = src_img_y

                matching_feature_count = 1
                while num_matches > 1:
                    img_id = int(data[5 + matching_feature_count])
                    img_id_x = data[6 + matching_feature_count]
                    img_id_y = data[7 + matching_feature_count]

                    matching_feature_count += 3

                    num_matches -= 1

                    x_row[0, img_id - 1] = img_id_x
                    y_row[0, img_id - 1] = img_id_y

                feature_x.append(x_row)
                feature_y.append(y_row)

        feature_x = np.array(feature_x).reshape(-1, num_images)
        feature_y = np.array(feature_y).reshape(-1, num_images)

        x_features.append(feature_x)
        y_features.append(feature_y)

    return x_features, y_features


def get_fundamental_matrices(x_features, y_features, num_images):

    f_matrix_list = list()
    inliers_list = list()

    for i in range(0, num_images - 1):
        x_feat = x_features[i]
        y_feat = y_features[i]

        for j in range(i + 1, num_images):

            img_i_feat_x = list()
            img_i_feat_y = list()
            img_j_feat_x = list()
            img_j_feat_y = list()

            for idx, x in enumerate(x_feat[:, j]):
                if x != -1.0:
                    img_j_feat_x.append(x)
                    img_j_feat_y.append(y_feat[:, j][idx])
                    img_i_feat_x.append(x_feat[:, i][idx])
                    img_i_feat_y.append(y_feat[:, i][idx])

            img_i_feat_x = np.array(img_i_feat_x).reshape(-1, 1)
            img_i_feat_y = np.array(img_i_feat_y).reshape(-1, 1)
            img_j_feat_x = np.array(img_j_feat_x).reshape(-1, 1)
            img_j_feat_y = np.array(img_j_feat_y).reshape(-1, 1)

            if img_i_feat_x.shape[0]:
                pts1 = np.hstack((img_i_feat_x, img_i_feat_y))
                pts2 = np.hstack((img_j_feat_x, img_j_feat_y))

                f_mat_obj = ir.FundamentalMatrix()
                inliers, f = f_mat_obj.get_fundamental_matrix(pts1, pts2)

                f_matrix_list.append(f)
                inliers_list.append(inliers)

    return f_matrix_list, inliers_list


def get_essential_matrices(f_matrix_list, K):

    e_mat_obj = em.EssentialMatrix(K, K)
    e_matrix_list = []

    for i in range(1):
        e = e_mat_obj.get_essential_matrix(f_matrix_list[i])
        e_matrix_list.append(e)

    return e_matrix_list


def get_camera_poses(e_matrix_list):

    R_list = list()
    C_list = list()

    for i in range(1):
        R, C = ep.extract_camera_poses(e_matrix_list[i])
        R_list.append(R)
        C_list.append(C)

    return R_list, C_list


def get_pts_3D(K, R_list, C_list, pts_list, linear=True, pts_3D_list=None):

    if linear:
        pts_3D_list = list()

        for idx in range(1):
            R = R_list[idx]
            C = C_list[idx]
            pts1 = pts_list[idx][0]
            pts2 = pts_list[idx][1]

            R1 = np.identity(3)
            C1 = np.zeros((3, 1))
            I = np.identity(3)

            P1 = np.dot(K, np.dot(R1, np.hstack((I, -C1))))
            pts_3D = list()

            for i in range(len(C)):
                p1 = pts1
                p2 = pts2

                R2 = R[i]
                C2 = C[i].reshape(3, 1)
                P2 = np.dot(K, np.dot(R2, np.hstack((I, -C2))))

                p3D = lt.linear_triangulation(P1, P2, p1, p2)
                p3D = p3D / p3D[:, 3].reshape(-1, 1)

                pts_3D.append(p3D)
            pts_3D_list.append(pts_3D)

        return pts_3D_list

    else:

        pts_3D_refined = list()

        for idx in range(1):
            R2 = R_list[idx]
            C2 = C_list[idx].reshape(-1, 1)
            pts1 = pts_list[idx][0]
            pts2 = pts_list[idx][1]

            R1 = np.identity(3)
            C1 = np.zeros((3, 1))
            I = np.identity(3)

            P1 = np.dot(K, np.dot(R1, np.hstack((I, -C1))))
            P2 = np.dot(K, np.dot(R2, np.hstack((I, -C2))))

            pts_3D = pts_3D_list[idx]
            pt_3D_refined = list()

            for i in range(len(pts_3D)):
                opt = scipy.optimize.least_squares(fun=reprojection_loss, x0=pts_3D[i], method="trf", args=[pts1[i], pts2[i], P1, P2], verbose=2)
                pt_3D_refined.append(opt.x)

            pts_3D_refined.append(np.array(pt_3D_refined))

        return pts_3D_refined

def get_camera_pose(pts_3d, pts_2d, K):
    C, R = PnPRANSAC(pts_3d, pts_2d, K)

def reprojection_loss(pt_3D, pt_1, pt_2, P1, P2):

    pt_1 = pt_1.reshape(-1, 1)
    pt_2 = pt_2.reshape(-1, 1)
    pt_3D = pt_3D.reshape(-1, 1)

    pt_1_projection = np.dot(P1, pt_3D)
    pt_1_projection /= pt_1_projection[-1]
    pt_1_reprojection_error = np.square(pt_1_projection[0] - pt_1[0]) + np.square(pt_1_projection[1] - pt_1[1])

    pt_2_projection = np.dot(P2, pt_3D)
    pt_2_projection /= pt_2_projection[-1]
    pt_2_reprojection_error = np.square(pt_2_projection[0] - pt_2[0]) + np.square(pt_2_projection[1] - pt_2[1])

    reprojection_loss = pt_1_reprojection_error + pt_2_reprojection_error

    final_loss = reprojection_loss[0].item()

    return final_loss


def get_best_R_and_C_and_pts_3D(pts_3D_list, R_list, C_list):

    best_R_list = list()
    best_C_list = list()
    best_pts_3D_list = list()

    for idx in range(1):
        R = R_list[idx]
        C = C_list[idx]
        pts_3D = pts_3D_list[idx]

        best_R, best_C, best_pts_3D = dc.disambiguate_camera_pose(R, C, pts_3D)

        best_R_list.append(best_R)
        best_C_list.append(best_C)
        best_pts_3D_list.append(best_pts_3D)

    return best_R_list, best_C_list, best_pts_3D_list


if __name__ == "__main__":

    calibration_file = "../Data/input/calibration.txt"
    K = read_calibration()
    print(K)

    num_images = 6
    images = read_images('../Data', num_images)

    a, b = get_matching_features('../Data', num_images)
