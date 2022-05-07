import numpy as np
from tqdm import tqdm


class FundamentalMatrix:
    def __init__(self, iterations=2000, threshold=0.001):
        self.iterations = iterations
        self.threshold = threshold
        self.epsilon = 1e-6

    def _get_A_matrix(self, img1_points, img2_points):

        A_mat = np.ndarray((img1_points.shape[0], 9))
        i = 0
        while i < img1_points.shape[0]:
            x1, y1 = img1_points[i][0], img1_points[i][1]
            x2, y2 = img2_points[i][0], img2_points[i][1]
            A_mat[i] = np.asarray([x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1])
            i += 1
        return A_mat

    def _compute_fundamental_matrix(self, points1, points2):

        A_matrix = self._get_A_matrix(points1, points2)
        U, S, V_T = np.linalg.svd(A_matrix)
        last_row = V_T[-1, :]
        F = np.reshape(last_row, (3, 3))

        U_, S_, V_T_ = np.linalg.svd(F)
        S_diag = np.diag(S_)
        S_diag[2, 2] = 0
        F = np.matmul(U_, np.dot(S_diag, V_T_))

        return F

    def _normalize(self, points):
        """
        https://www.cc.gatech.edu/classes/AY2016/cs4476_fall/results/proj3/html/sdai30/index.html
        """
        points_ = np.mean(points, axis=0)
        x_, y_ = points_[0], points_[1]
        x_cap, y_cap = points[:, 0] - x_, points[:, 1] - y_

        s = (2 / np.mean(x_cap ** 2 + y_cap ** 2)) ** (0.5)
        T_scale = np.diag([s, s, 1])
        T_trans = np.array([[1, 0, -x_], [0, 1, -y_], [0, 0, 1]])
        T = T_scale.dot(T_trans)

        pts_ = np.column_stack((points, np.ones(len(points))))
        pts_norm = (T.dot(pts_.T)).T

        return pts_norm, T

    def get_fundamental_matrix(self, img1_points_, img2_points_, pt_pair_idx):

        pt_pair_idx = np.array(pt_pair_idx).reshape(-1)

        max_inlier_count = 0
        max_inlier_indices = []
        max_inlier_f = None

        num_points = img1_points_.shape[0]

        img1_points, T1 = self._normalize(img1_points_)
        img2_points, T2 = self._normalize(img2_points_)

        for i in tqdm(range(self.iterations), desc='RANSAC going on'):

            indices = np.random.choice(num_points, size=8)

            img1_random_points = img1_points[indices]
            img2_random_points = img2_points[indices]

            F = self._compute_fundamental_matrix(img1_random_points, img2_random_points)
            F = np.matmul(T2.T, np.matmul(F, T1))
            F /= F[2, 2]

            inlier_indices = list()
            for n in range(num_points):
                error = get_epipolar_constraint_error(F, img1_points_[n, :], img2_points_[n, :])
                if error < self.threshold:
                    inlier_indices.append(pt_pair_idx[n])

            if len(inlier_indices) > max_inlier_count:
                max_inlier_count = len(inlier_indices)
                max_inlier_indices = inlier_indices
                max_inlier_f = F

        best_fundamental_matrix = max_inlier_f

        return best_fundamental_matrix, max_inlier_indices


def get_epipolar_constraint_error(fundamental_matrix, pts1, pts2):
    pts1_homo = np.array([pts1[0], pts1[1], 1])
    pts2_homo = np.array([pts2[0], pts2[1], 1])

    epipolar_constraint_error = np.dot(pts2_homo.T, np.dot(fundamental_matrix, pts1_homo))
    epipolar_error = np.abs(epipolar_constraint_error)

    return epipolar_error
