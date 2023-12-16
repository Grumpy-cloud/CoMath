import pprint

import matplotlib.pyplot as plt
import numpy as np

from Lab2.solver import MatrixSolver


def build_plot(residuals: np.array) -> None:
    plt.title('')

    plt.scatter(
        x=np.array([i for i in range(len(residuals))]),
        y=residuals
    )

    plt.xlabel(r'$ k $')
    plt.ylabel(r'$ ||r|| $')

    plt.grid(True)

    plt.show()


def create_matrix() -> np.matrix:
    n = 100
    a = 1
    b = 10
    c = 1
    p = 1

    matrix = np.matrix(np.zeros((n, n + 1)))

    j = -1
    for i in range(n - 1):
        if j >= 0:
            matrix[i, j] = a

        matrix[i, j + 1] = b
        matrix[i, j + 2] = c

        j += 1

    for j in range(n):
        matrix[-1, j] = 1

    for i in range(n):
        matrix[i, -1] = i + 1

    return matrix


def solve_by_direct_method(
        msg: str,
        method
) -> None:
    x, r = method()

    print(msg)
    pprint.pprint(x)
    print('residual: ', r, '\n')


def solve_by_iterative_method(
        msg: str,
        method
) -> None:
    x, r = method()

    print(msg)
    pprint.pprint(x)
    build_plot(r)
    print()


if __name__ == '__main__':
    matrix = create_matrix()

    matrix_solver = MatrixSolver(matrix)

    solve_by_direct_method(
        'Gauss method with main element selection:',
        matrix_solver.solve_by_gauss_method_with_main_element_selection
    )

    solve_by_direct_method(
        'LU method:',
        matrix_solver.solve_by_LU_method
    )

    solve_by_iterative_method(
        'Seidel method:',
        matrix_solver.solve_by_seidel_method
    )

    solve_by_iterative_method(
        'Jacobi method:',
        matrix_solver.solve_by_jacobi_method
    )

    solve_by_iterative_method(
        'Upper relaxation method:',
        matrix_solver.solve_by_upper_relaxation_method
    )
