import numpy as np


class EssentialMatrix:
    def __init__(self, K1, K2):
        self.K1 = K1
        self.K2 = K2

    def get_essential_matrix(self, fundamental_matrix):
        E = np.matmul(self.K2.T, np.matmul(fundamental_matrix, self.K1))
        U, S, V_T = np.linalg.svd(E)
        S = [1, 1, 0]
        E = np.matmul(U, np.dot(np.diag(S), V_T))

        return E
