import numpy as np


def extract_camera_poses(e_matrix):

    U, S, V_T = np.linalg.svd(e_matrix)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)

    C1 = U[:, 2]
    C2 = -U[:, 2]
    C3 = U[:, 2]
    C4 = -U[:, 2]
    R1 = np.matmul(U, np.matmul(W, V_T))
    R2 = np.matmul(U, np.matmul(W, V_T))
    R3 = np.matmul(U, np.matmul(W.T, V_T))
    R4 = np.matmul(U, np.matmul(W.T, V_T))

    R = np.array([R1, R2, R3, R4])
    C = np.array([C1, C2, C3, C4])

    for i in range(4):
        if np.linalg.det(R[i]) < 0:
            R[i] = -R[i]
            C[i] = -C[i]

    return R, C