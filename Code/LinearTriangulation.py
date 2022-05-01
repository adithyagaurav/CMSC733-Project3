import numpy as np


def linear_triangulation(P1, P2, pts1, pts2):
    """
    http://www.cs.cmu.edu/~16385/s17/Slides/11.4_Triangulation.pdf
    """
    p1_T = P1[0, :].reshape(1, 4)
    p2_T = P1[1, :].reshape(1, 4)
    p3_T = P1[2, :].reshape(1, 4)

    p1_dash_T = P2[0, :].reshape(1, 4)
    p2_dash_T = P2[1, :].reshape(1, 4)
    p3_dash_T = P2[2, :].reshape(1, 4)

    pts_3D = list()

    for i in range(pts1.shape[0]):

        x, y = pts1[i, 0], pts1[i, 1]
        x_dash, y_dash = pts2[i, 0], pts2[i, 1]

        A = np.array([[y * p3_T - p2_T],
                      [p1_T - x * p3_T],
                      [y_dash * p3_dash_T - p2_dash_T],
                      [p1_dash_T - x_dash * p3_dash_T]])

        A = np.reshape(A, (4, 4))
        _, _, V_T = np.linalg.svd(A)
        x = V_T[-1, :]
        pts_3D.append(x)

    return np.array(pts_3D)

