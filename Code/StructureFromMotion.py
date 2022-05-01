import Utils as ut


def run(input_dir, output_dir, num_images):

    images = ut.read_images(input_dir, num_images)

    x_features, y_features = ut.get_matching_features(input_dir, num_images)

    f_matrix_list, inliers_list = ut.get_fundamental_matrices(x_features, y_features, num_images)

    K = ut.read_calibration()

    # only for first two images
    e_matrix_list = ut.get_essential_matrices(f_matrix_list, K)

    # only for first two images
    r_list, c_list = ut.get_camera_poses(e_matrix_list)

    # only for first two images
    pts_3d_list = ut.get_pts_3D(K, r_list, c_list, inliers_list, linear=True, pts_3D_list=None)

    # only for first two images
    best_r_list, best_c_list, best_pts_3D_list = ut.get_best_R_and_C_and_pts_3D(pts_3d_list, r_list, c_list)

    # only for first two images
    pts_3d_list_refined = ut.get_pts_3D(K, best_r_list, best_c_list, inliers_list, linear=False, pts_3D_list=best_pts_3D_list)

    # now processing over images 3 - 6, check section 4: project overview


if __name__ == "__main__":
    input_dir = "../Data/input"
    output_dir = ""
    num_images = 6

    run(input_dir, output_dir, num_images)
