import numpy as np
import random

def update_A(A, pt1, pt2):
    x, y, _ = pt2
    A.append(np.hstack((np.hstack((np.zeros(1,4), -1*pt1)), y*pt1)))
    A.append(np.hstack((np.hstack((pt1, np.zeros(1,4))), -1*x*pt1)))
    A.append(np.hstack((np.hstack((-1*y*pt1, x*pt1)), np.zeros(1,4))))
    return A

def do_svd(X, Y, K_inv):
    X_homo = np.hstack((X, np.ones(len(X), 1)))
    Y_homo = np.hstack((Y, np.ones(len(Y), 1)))

    Y_trans = np.dot(K_inv, Y_homo.T).T
    A = []
    for i in range(len(X)):
        pt_3d, pt_2d = X_homo[i], Y_trans[i]
        A = update_A(A, pt_3d, pt_2d)
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

def do_reprojection(pts_3d, K, R, C):
    pts_3d_homo = np.hstack((pts_3d, np.ones(len(pts_3d), 1)))
    P = np.dot(np.dot(K, R), np.hstack((np.identity(3), -C)))
    pts_2d_homo = np.dot(P, pts_3d_homo)
    pts_2d = pts_2d_homo/pts_2d_homo[2][0]
    return pts_2d[:-1].T

def get_error(pt1, pt2):
    return np.sqrt((pt1[0]-pt2[0])**2, (pt1[1]-pt1[1])**2)

def run_linear_pnp(pts_3d, pts_2d, K):
    best_score = 0
    threshold = 5
    n_iters = 1000
    # pts_2d_homo = np.hstack((pts_2d, np.ones(len(pts_2d), 1)))
    #
    # C = np.array([[0],[0],[0]])
    # R = np.array([[1,0,0],[0,1,0],[0,0,1]])

    for i in range(n_iters):
        rand_idxs = list(np.random.randint(low = 0,high=len(pts_3d),size=6))
        rand_pts3d, rand_pts2d = pts_3d[rand_idxs], pts_2d[rand_idxs]
        R, C = do_svd(rand_pts3d, rand_pts2d)
        score = 0
        for j in range(len(pts_2d)):
            reproj_pt = do_reprojection(pts_3d, K, R, C)
            error = get_error(reproj_pt, pts_2d[i])
            if error<threshold:
                score+=1
        if score>best_score:
            best_score = score
            final_R = R
            final_C = C

    return final_R, final_C
