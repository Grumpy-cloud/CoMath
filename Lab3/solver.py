from typing import Tuple, Callable, Union

import matplotlib.pyplot as plt
import numpy as np


class Solver:
    @staticmethod
    def solve_equation_by_simple_iteration_method(
            f: Callable[[float], float],
            x0: float,
            eps: float = 1e-5
    ) -> Tuple[float, np.array]:
        residuals = []

        current_x = x0
        next_x = f(current_x)

        residuals.append(abs(next_x - current_x))

        while residuals[-1] > eps:
            current_x = next_x
            next_x = f(current_x)

            residuals.append(abs(next_x - current_x))

        return next_x, np.array(residuals)

    @staticmethod
    def solve_equation_by_newton_method(
            F: Callable[[float], float],
            x0: float,
            eps: float = 1e-5
    ) -> Tuple[float, np.array]:

        def derivative(
                x: float,
                h: float = 1e-2
        ) -> float:
            return (F(x + h) - F(x)) / h

        residuals = []

        current_x = x0
        next_x = current_x - derivative(current_x) ** (-1) * F(current_x)

        residuals.append(abs(next_x - current_x))

        while residuals[-1] > eps:
            current_x = next_x
            next_x = current_x - derivative(current_x) ** (-1) * F(current_x)

            residuals.append(abs(next_x - current_x))

        return next_x, np.array(residuals)

    @staticmethod
    def solve_system_by_simple_iteration_method(
            f: np.array,
            u0: np.array,
            eps: float = 1e-5
    ) -> Tuple[np.array, np.array]:
        residuals = []

        current_u = u0
        next_u = np.array([f[i](*current_u) for i in range(len(f))])

        residuals.append(np.linalg.norm(next_u - current_u))

        while residuals[-1] > eps:
            current_u = next_u
            next_u = np.array([f[i](*current_u) for i in range(len(f))])

            residuals.append(np.linalg.norm(next_u - current_u))

        return next_u, np.array(residuals)

    @staticmethod
    def solve_system_by_newton_method(
            F: np.array,
            u0: np.array,
            eps: float = 1e-5
    ) -> Tuple[np.array, np.array]:
        n = u0.shape[0]

        def calculate_jacobi_matrix(u: np.array) -> np.matrix:

            def derivative(
                    i: int,
                    j: int,
                    h: float = 1e-2
            ) -> float:
                return (F[i](*(u + np.array([h if index == j else 0 for index in range(n)]))) - F[i](*u)) / h

            J = np.zeros((n, n))

            for i in range(n):
                for j in range(n):
                    J[i, j] = derivative(i, j)

            return J

        residuals = []

        current_u = u0
        next_u = current_u - np.matmul(np.linalg.inv(calculate_jacobi_matrix(current_u)), np.array([F[i](*current_u) for i in range(n)]))

        residuals.append(np.linalg.norm(next_u - current_u))

        while residuals[-1] > eps:
            current_u = next_u
            next_u = current_u - np.matmul(np.linalg.inv(calculate_jacobi_matrix(current_u)), np.array([F[i](*current_u) for i in range(n)]))

            residuals.append(np.linalg.norm(next_u - current_u))

        return next_u, np.array(residuals)

    @staticmethod
    def build_plot(
            x: Union[float, np.array],
            residuals: np.array,
            symbol: str
    ) -> None:
        plt.title(str(symbol) + ' = ' + repr(x))

        plt.scatter(
            x=1 + np.arange(len(residuals)),
            y=residuals
        )

        plt.xlabel(r'$ k $')
        plt.ylabel(r'$ ||r|| $')

        plt.grid(True)

        plt.show()
