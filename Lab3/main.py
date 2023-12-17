import numpy as np

from Lab3.solver import Solver

if __name__ == '__main__':
    x, residuals = Solver.solve_equation_by_simple_iteration_method(
        f=lambda x: (3 - 2 * x * x) / 5,
        x0=0
    )
    Solver.build_plot(x, residuals, 'x')

    x, residuals = Solver.solve_equation_by_newton_method(
        F=lambda x: 2 * x*x + 5*x - 3,
        x0=-1.3
    )
    Solver.build_plot(x, residuals, 'x')

    x, residuals = Solver.solve_system_by_simple_iteration_method(
        f=np.array([lambda x, y: 1 - np.cos(y) / 2, lambda x, y: np.sin(x + 1) - 12]),
        u0=np.array([0, 0]),
        eps=1e-3
    )
    Solver.build_plot(x, residuals, '(x, y)')

    x, residuals = Solver.solve_system_by_newton_method(
        F=np.array([lambda x, y: np.sin(x + 1) - y - 12, lambda x, y: np.cos(y) + 2 * x - 2]),
        u0=np.array([0, 0]),
        eps=1e-3
    )
    Solver.build_plot(x, residuals, '(x, y)')
