from typing import Tuple

import numpy as np


class MatrixSolver:
    @staticmethod
    def _calculate_A_and_f_from_matrix_coefficients(
            matrix: np.matrix
    ) -> Tuple[np.matrix, np.matrix]:
        n = matrix.shape[0]
        m = matrix.shape[1] - 1

        A = np.matrix(np.zeros([n, m]))
        f = np.matrix(np.zeros([n, 1]))

        for i in range(n):
            for j in range(m):
                A[i, j] = matrix[i, j]

            f[i, 0] = matrix[i, -1]

        return A, f

    def __init__(
            self,
            matrix: np.matrix
    ) -> None:
        self._matrix = matrix
        self._A, self._f = self._calculate_A_and_f_from_matrix_coefficients(matrix)

    def _calculate_difference(
            self,
            x: np.matrix
    ) -> np.matrix:
        return self._A * x - self._f

    def solve_by_gauss_method_with_main_element_selection(self) -> Tuple[np.matrix, float]:
        matrix = self._matrix.copy()

        n = matrix.shape[0]

        for i in range(n):
            max_elem = abs(matrix[i, i])
            max_row = i

            for k in range(i + 1, n):
                if abs(matrix[k, i]) > max_elem:
                    max_elem = abs(matrix[k, i])
                    max_row = k

            row = np.copy(matrix[i])
            matrix[i] = matrix[max_row]
            matrix[max_row] = row

            for k in range(i + 1, n):
                factor = matrix[k, i] / matrix[i, i]

                for j in range(i, n + 1):
                    if i == j:
                        matrix[k, j] = 0
                    else:
                        matrix[k, j] -= factor * matrix[i, j]

        x = np.matrix(np.zeros([n, 1]))

        for i in range(n - 1, -1, -1):
            x[i, 0] = matrix[i, n] / matrix[i, i]

            for k in range(i - 1, -1, -1):
                matrix[k, n] -= matrix[k, i] * x[i, 0]

        return x, np.linalg.norm(self._calculate_difference(x))

    def _decompose_to_LU(self) -> np.matrix:
        A = self._A.copy()

        n = A.shape[0]

        lu_matrix = np.matrix(np.zeros([n, n]))

        for k in range(n):
            for j in range(k, n):
                lu_matrix[k, j] = A[k, j] - lu_matrix[k, : k] * lu_matrix[: k, j]

            for i in range(k + 1, n):
                lu_matrix[i, k] = (A[i, k] - lu_matrix[i, :k] * lu_matrix[:k, k]) / lu_matrix[k, k]

        return lu_matrix

    def solve_by_LU_method(self) -> Tuple[np.matrix, float]:
        lu_matrix = self._decompose_to_LU()
        f = self._f

        y = np.matrix(np.zeros([lu_matrix.shape[0], 1]))

        for i in range(y.shape[0]):
            y[i, 0] = f[i] - lu_matrix[i, :i] * y[:i]

        x = np.matrix(np.zeros([lu_matrix.shape[0], 1]))

        for i in range(1, x.shape[0] + 1):
            x[-i, 0] = (y[-i] - lu_matrix[-i, -i:] * x[-i:, 0]) / lu_matrix[-i, -i]

        return x, np.linalg.norm(self._calculate_difference(x))

    def solve_by_seidel_method(
            self,
            eps: float = 1e-5
    ) -> Tuple[np.matrix, np.array]:
        A = self._A
        f = self._f

        n = A.shape[0]

        x = np.zeros_like(f)
        residuals = []

        while True:
            old_x = x.copy()

            for i in range(n):
                temp1 = A[i, :i] * x[:i]
                temp2 = A[i, i + 1:] * x[i + 1:]
                x[i] = (f[i] - temp1 - temp2) / A[i, i]

            residuals.append(np.linalg.norm(self._calculate_difference(x)))

            if np.linalg.norm(x - old_x) < eps:
                return x, np.array(residuals)

    def solve_by_jacobi_method(
            self,
            eps: float = 1e-5
    ) -> Tuple[np.matrix, np.array]:
        A = self._A
        f = self._f

        n = A.shape[0]

        x = np.zeros_like(f)
        residuals = []

        D = np.matrix(np.zeros(A.shape))

        for i in range(n):
            D[i, i] = A[i, i]

        R = A - D

        while True:
            old_x = np.copy(x)

            x = np.linalg.inv(D) * (f - R * x)

            residuals.append(np.linalg.norm(self._calculate_difference(x)))

            if np.linalg.norm(x - old_x) < eps:
                return x, np.array(residuals)

    def solve_by_upper_relaxation_method(
            self,
            omega: float = 1.1,
            eps: float = 1e-6
    ) -> Tuple[np.matrix, np.array]:
        A = self._A
        f = self._f

        n = A.shape[0]

        x = np.zeros_like(f)
        residuals = []

        while True:
            x_old = x.copy()

            for i in range(n):
                row_sum = A[i, :i] * x[:i] + A[i, i + 1:] * x[i + 1:]
                x[i] = (1 - omega) * x_old[i] + omega * (f[i] - row_sum) / A[i, i]

            residuals.append(np.linalg.norm(self._calculate_difference(x)))

            if np.linalg.norm(x - x_old) < eps:
                return x, np.array(residuals)
