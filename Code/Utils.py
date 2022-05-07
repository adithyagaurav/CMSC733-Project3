import os
import cv2
import numpy as np
import scipy.optimize

import GetInliersRANSAC as ir
import EssentialMatrixFromFundamentalMatrix as em
import ExtractCameraPose as ep
import LinearTriangulation as lt
import DisambiguateCameraPose as dc


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

        file_data = match_file_data.readlines()
        for idx in range(1, len(file_data)):
            line = file_data[idx]
            row_data = line.split()
            data = list(float(val) for val in row_data)
            data = np.array(data)

            num_matches = data[0]

            src_img_x, src_img_y = data[4], data[5]

            x_coords = np.ones((1, num_images)) * (-1)
            y_coords = np.ones((1, num_images)) * (-1)

            x_coords[0, i-1] = src_img_x
            y_coords[0, i-1] = src_img_y

            matching_feature_count = 1

            while num_matches > 1:
                img_id = int(data[5 + matching_feature_count])
                img_id_x = data[6 + matching_feature_count]
                img_id_y = data[7 + matching_feature_count]
                x_coords[0, img_id - 1] = img_id_x
                y_coords[0, img_id - 1] = img_id_y

                matching_feature_count += 3
                num_matches -= 1

            x_features.append(x_coords)
            y_features.append(y_coords)

    x_features = np.array(x_features).reshape(-1, num_images)
    y_features = np.array(y_features).reshape(-1, num_images)

    return x_features, y_features


def get_inliers(x_features, y_features, num_images, images, output_dir):

    inlier_idx_flag = np.zeros_like(x_features)
    f_matrices = np.zeros((num_images, num_images), dtype=object)
    f_mat_obj = ir.FundamentalMatrix()
    # for i in range(0, num_images - 1):
    for i in range(0, 1):
        img1 = images[i]
        h, w = img1.shape[:2]
        for j in range(i + 1, num_images):
            img2 = images[j]
            stacked_img = np.hstack((img1, img2))

            pt_pair_idx = np.where((x_features[:, i] != -1) & (x_features[:, j] != -1))
            pts1 = np.hstack((x_features[pt_pair_idx, i].reshape(-1, 1), y_features[pt_pair_idx, i].reshape(-1, 1)))
            pts2 = np.hstack((x_features[pt_pair_idx, j].reshape(-1, 1), y_features[pt_pair_idx, j].reshape(-1, 1)))

            if pts1.shape[0] > 8:
                print('[INFO]: Computing Inliers For Image {0} and Image {1}'.format(i, j))
                f, inlier_idx = f_mat_obj.get_fundamental_matrix(pts1, pts2, pt_pair_idx)
                print('[INFO]: Fundamental Matrix for Image {0} and Image {1} \n {2}'.format(i, j, f))
                f_matrices[i, j] = f
                inlier_idx_flag[inlier_idx, i] = 1
                inlier_idx_flag[inlier_idx, j] = 1

                p1 = pts1
                p2 = pts2
                for pt1, pt2 in zip(p1, p2):
                    pt2[0] += w
                    cv2.line(stacked_img, np.int32(pt1), np.int32(pt2), (0, 0, 255), 1, cv2.LINE_AA)

                inlier_idx_flag_ = np.int32(inlier_idx_flag)
                idx = np.where(inlier_idx_flag_[:, i] & inlier_idx_flag_[:, j])
                img1_pts = np.hstack((x_features[idx, i].reshape(-1, 1), y_features[idx, i].reshape(-1, 1)))
                img2_pts = np.hstack((x_features[idx, j].reshape(-1, 1), y_features[idx, j].reshape(-1, 1)))

                for pt1, pt2 in zip(img1_pts, img2_pts):
                    pt2[0] += w
                    cv2.line(stacked_img, np.int32(pt1), np.int32(pt2), (0, 255, 0), 1, cv2.LINE_AA)

            out_dir = os.path.join(output_dir, "FeatureCorrespondenceOutputForAllImageSet")
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            # cv2.imwrite(os.path.join(out_dir, "img_{0}_{1}_inlier_outlier.png".format(i + 1, j + 1)), stacked_img)

    return f_matrices, inlier_idx_flag


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

    for i in range(len(e_matrix_list)):
        R, C = ep.extract_camera_poses(e_matrix_list[i])
        R_list.append(R)
        C_list.append(C)

    return R_list, C_list


def get_pts_3D(K, R_list, C_list, pts_list, linear=True, pts_3D_list=None):

    if linear:
        print('[INFO]: Started Linear Triangulation')
        pts_3D_list = list()

        # import matplotlib.pyplot as plt

        for idx in range(len(R_list)):
            R = R_list[idx]
            C = C_list[idx]
            pts1 = pts_list[idx][0]
            pts2 = pts_list[idx][1]

            R1 = np.identity(3)
            C1 = np.zeros((3, 1))
            I = np.identity(3)

            P1 = np.dot(K, np.dot(R1, np.hstack((I, -C1))))
            pts_3D = list()

            # colors = ['b', 'r', 'o', 'g']
            for i in range(len(C)):
                p1 = pts1
                p2 = pts2

                R2 = R[i]
                C2 = C[i].reshape(3, 1)
                P2 = np.dot(K, np.dot(R2, np.hstack((I, -C2))))

                p3D = lt.linear_triangulation(P1, P2, p1, p2)
                p3D = p3D / p3D[:, 3].reshape(-1, 1)

                # color = colors[i]
                # plt.plot(p3D[:, 0], p3D[:, 2], '.', color)

                pts_3D.append(p3D)
            # plt.show()
            pts_3D_list.append(pts_3D)

        return pts_3D_list

    else:
        print('[INFO]: Started Non-Linear Triangulation')
        pts_3D_refined = list()

        for idx in range(len(R_list)):
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
                opt = scipy.optimize.least_squares(fun=reprojection_loss, x0=pts_3D[i], method="trf", args=[pts1[i], pts2[i], P1, P2])
                pt_3D_refined.append(opt.x)

            pts_3D_refined.append(np.array(pt_3D_refined))

        return pts_3D_refined


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

    final_loss = reprojection_loss.squeeze()

    return final_loss

def get_best_R_and_C_and_pts_3D(pts_3D_list, R_list, C_list):

    best_R_list = list()
    best_C_list = list()
    best_pts_3D_list = list()

    for idx in range(len(pts_3D_list)):
        R = R_list[idx]
        C = C_list[idx]
        pts_3D = pts_3D_list[idx]

        best_R, best_C, best_pts_3D = dc.disambiguate_camera_pose(R, C, pts_3D)

        best_R_list.append(best_R)
        best_C_list.append(best_C)
        best_pts_3D_list.append(best_pts_3D)

    print("best R", best_R_list)
    print("best C", best_C_list)

    return best_R_list, best_C_list, best_pts_3D_list


def get_mean_reprojection_error(K, pts_3D, pts1, pts2, R1, C1, R2, C2):

    I = np.identity(3)
    if R1 is None:
        R1 = np.identity(3)
        C1 = np.zeros((3, 1))

    C1 = C1.reshape(-1, 1)
    C2 = C2.reshape(-1, 1)

    P1 = np.dot(K, np.dot(R1, np.hstack((I, -C1))))
    P2 = np.dot(K, np.dot(R2, np.hstack((I, -C2))))

    mean_loss = list()

    for i in range(len(pts_3D)):
        rep_error = reprojection_loss(pts_3D[i], pts1[i], pts2[i], P1, P2)
        mean_loss.append(rep_error)

    return np.mean(mean_loss)


if __name__ == "__main__":

    calibration_file = "../Data/input/calibration.txt"
    K = read_calibration()
    print(K)

    num_images = 6
    images = read_images('../Data', num_images)

    a, b = get_matching_features('../Data', num_images)
