import numpy as np


def update_A(A, pt1, pt2):
    x, y, _ = pt2
    pt1_ = np.array(pt1).reshape((1,4))
    A.append(np.hstack((np.hstack((np.zeros((1,4)), -1*pt1_)), y*pt1_)))
    A.append(np.hstack((np.hstack((pt1_, np.zeros((1,4)))), -1*x*pt1_)))
    A.append(np.hstack((np.hstack((-1*y*pt1_, x*pt1_)), np.zeros((1,4)))))
    return A

def do_svd(X, Y, K_inv):
    X_homo = np.hstack((X, np.ones((X.shape[0], 1))))
    Y_homo = np.hstack((Y, np.ones((len(Y), 1))))

    Y_trans = np.dot(K_inv, Y_homo.T).T
    A = []
    for i in range(len(X)):
        pt_3d, pt_2d = X_homo[i].reshape((1,4)), Y_trans[i].reshape((1,3))
        u_cross = np.array([[0, -1, pt_2d[0][1]],
                            [1, 0, -pt_2d[0][0]],
                            [-pt_2d[0][1], pt_2d[0][0], 0]])
        X_tilde = np.vstack((np.hstack((  pt_3d, np.zeros((1, 4)), np.zeros((1, 4)))),
                            np.hstack((np.zeros((1, 4)),     pt_3d, np.zeros((1, 4)))),
                            np.hstack((np.zeros((1, 4)), np.zeros((1, 4)),     pt_3d))))
        a = u_cross.dot(X_tilde)
        if i>0:
            A = np.vstack((A, a))
        else:
            A = a
    A = np.array(A)
    U, S, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape((3,4))
    R = P[:, :3]
    C = P[:, -1]
    U, S, Vt = np.linalg.svd(R)
    S = np.array([[1,0,0],[0,1,0],[0,0,np.linalg.det(np.dot(U, Vt))]])
    R = np.dot(np.dot(U, S), Vt)
    C = -1 * np.dot(np.linalg.inv(R), C)
    if np.linalg.det(R)<0:
        return -1*R, -1*C
    return R, C


def homo(pts):
    return np.hstack((pts, np.ones((pts.shape[0], 1))))


def do_reprojection(pts_3d, K, R, C):
    pts_3d_ = np.array(pts_3d).reshape((1,3))
    C_ = C.reshape((3,1))
    pts_3d_homo = np.hstack((pts_3d_, np.ones((len(pts_3d_), 1)))).reshape((4,1))
    P = np.dot(np.dot(K, R), np.hstack((np.identity(3), -C_)))
    pts_2d_homo = np.dot(P, pts_3d_homo)
    pts_2d = pts_2d_homo/pts_2d_homo[2][0]
    return pts_2d[:-1].T


def get_error(pt1, pt2):
    return np.sqrt((pt1[0][0]-pt2[0])**2 + (pt1[0][1]-pt2[1])**2)


def reprojectionErrorPnP(pts_3D, pts_2D, K, R, C):
    I = np.eye(3)
    C = C.reshape(-1, 1)
    P = np.dot(K, np.dot(R, np.hstack((I, -C))))
    error = list()

    for pt_3D, pt_2D in zip(pts_3D, pts_2D):
        p_1T, p_2T, p_3T = P  # rows of P
        p_1T, p_2T, p_3T = p_1T.reshape(1, -1), p_2T.reshape(1, -1), p_3T.reshape(1, -1)
        X = pt_3D.reshape(1, -1)
        pt_3D_ = np.hstack((X, np.ones((X.shape[0], 1)))).reshape(-1, 1)

        pt2D_x_proj = np.divide(p_1T.dot(pt_3D_), p_3T.dot(pt_3D_))
        pt2D_y_proj = np.divide(p_2T.dot(pt_3D_), p_3T.dot(pt_3D_))

        reproj_error = np.square(pt_2D[1] - pt2D_y_proj) + np.square(pt_2D[0] - pt2D_x_proj)

        error.append(reproj_error)

    return np.sqrt(np.mean(error)).squeeze()


def PnPRANSAC(K, pts_2d, pts_3d):
    best_score = 0
    threshold = 10
    n_iters = 1000
    final_R = None
    final_C = None
    for i in range(n_iters):
        rand_idxs = list(np.random.randint(low=0, high=len(pts_3d), size=6))
        rand_pts3d, rand_pts2d = pts_3d[rand_idxs], pts_2d[rand_idxs]
        R, C = do_svd(rand_pts3d, rand_pts2d, np.linalg.inv(K))
        score = 0
        for j in range(len(pts_2d)):
            reproj_pt = do_reprojection(pts_3d[j], K, R, C)
            error = get_error(reproj_pt, pts_2d[j])
            if error < threshold:
                score += 1
        if score > best_score:
            best_score = score
            final_R = R
            final_C = C

    if final_R is None:
        print('[WARN]: No PnP Match Found')

    return final_R, final_C