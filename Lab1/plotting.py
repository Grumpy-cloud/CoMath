import numpy as np
import matplotlib.pyplot as plt

from Lab1.error_calculations import MethodError


class BaseErrorPlots:
    def __init__(self, x: float) -> None:
        self.x0 = x
        self.step_arr = [2 / np.power(2, n) for n in range(1, 21)]

    def function(self, x: float) -> float:
        pass

    def first_derivative(self, x: float) -> float:
        pass

    def build_plot(self, file_name: str) -> None:
        error_calculator = MethodError(self.function, self.first_derivative, self.x0)

        one_point_method_err_values = [error_calculator.one_point_error(step) for step in self.step_arr]
        one_point_rev_method_err_values = [error_calculator.one_point_reversed_error(step) for step in self.step_arr]
        two_points_method_err_values = [error_calculator.two_points_error(step) for step in self.step_arr]
        four_points_method_err_values = [error_calculator.four_points_error(step) for step in self.step_arr]
        six_points_method_err_values = [error_calculator.six_points_error(step) for step in self.step_arr]

        plt.figure(figsize=(10, 10), dpi=200)
        plt.xscale('log')
        plt.yscale('log')
        plt.title('Method error dependency from step')
        plt.xlabel('Step')
        plt.ylabel('Absolute error')

        plt.plot(self.step_arr, one_point_method_err_values, 'sk', label='One point method')
        plt.plot(self.step_arr, one_point_rev_method_err_values, 'r^', label='Reversed one point method')
        plt.plot(self.step_arr, two_points_method_err_values, 'b.', label='Two points method')
        plt.plot(self.step_arr, four_points_method_err_values, 'gv', label='Four points method')
        plt.plot(self.step_arr, six_points_method_err_values, 'mp', label='Six points method')

        plt.legend()
        plt.savefig(file_name)
        plt.show()


class FirstFunctionErrorPlots(BaseErrorPlots):
    def __init__(self, x: float) -> None:
        super().__init__(x)

    def function(self, x: float) -> float:
        return np.sin(np.power(x, 2))

    def first_derivative(self, x: float) -> float:
        return 2 * x * np.cos(np.power(x, 2))


class SecondFunctionErrorPlots(BaseErrorPlots):
    def __init__(self, x: float) -> None:
        super().__init__(x)

    def function(self, x: float) -> float:
        return np.cos(np.sin(x))

    def first_derivative(self, x: float) -> float:
        return -(np.sin(np.sin(x)) * np.cos(x))


class ThirdFunctionErrorPlots(BaseErrorPlots):
    def __init__(self, x: float) -> None:
        super().__init__(x)

    def function(self, x: float) -> float:
        return np.exp(np.sin(np.cos(x)))

    def first_derivative(self, x: float) -> float:
        return -(np.sin(x) * np.cos(np.cos(x)) * np.exp(np.sin(np.cos(x))))


class FourthFunctionErrorPlots(BaseErrorPlots):
    def __init__(self, x: float) -> None:
        super().__init__(x)

    def function(self, x: float) -> float:
        return np.log(x + 3)

    def first_derivative(self, x: float) -> float:
        return 1 / (x + 3)


class FifthFunctionErrorPlots(BaseErrorPlots):
    def __init__(self, x: float) -> None:
        super().__init__(x)

    def function(self, x: float) -> float:
        return np.sqrt(x + 3)

    def first_derivative(self, x: float) -> float:
        return 1 / (2 * np.sqrt(x + 3))


if __name__ == '__main__':
    x0 = 10

    first_function_plots = FirstFunctionErrorPlots(x0)
    first_function_plots.build_plot('sin(x^2)')

    second_function_plots = SecondFunctionErrorPlots(x0)
    second_function_plots.build_plot('cos(sin(x))')

    third_function_plots = ThirdFunctionErrorPlots(x0)
    third_function_plots.build_plot('exp(sin(cos(x)))')

    fourth_function_plots = FourthFunctionErrorPlots(x0)
    fourth_function_plots.build_plot('ln(x + 3)')

    fifth_function_plots = FifthFunctionErrorPlots(x0)
    fifth_function_plots.build_plot('sqrt(x + 3)')
