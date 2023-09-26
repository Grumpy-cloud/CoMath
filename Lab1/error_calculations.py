import numpy as np


class MethodError:
    def __init__(
            self,
            func: callable,
            der: callable,
            x: float
    ) -> None:
        self.func = func
        self.der = der
        self.x0 = x

    def one_point_error(
            self,
            step: float
    ) -> float:
        first_member = (self.func(self.x0 + step) - self.func(self.x0)) / step
        abs_error = np.abs(self.der(self.x0) - first_member)
        return abs_error

    def one_point_reversed_error(
            self,
            step: float
    ) -> float:
        first_member = (self.func(self.x0) - self.func(self.x0 - step)) / step
        abs_error = np.abs(self.der(self.x0) - first_member)
        return abs_error

    def two_points_error(
            self,
            step: float
    ) -> float:
        first_member = (self.func(self.x0 + step) - self.func(self.x0 - step)) / (2 * step)
        abs_error = np.abs(self.der(self.x0) - first_member)
        return abs_error

    def four_points_error(
            self,
            step: float
    ) -> float:
        first_member = 4 * (self.func(self.x0 + step) - self.func(self.x0 - step)) / (6 * step)
        second_member = (self.func(self.x0 + 2 * step) - self.func(self.x0 - 2 * step)) / (12 * step)
        abs_error = np.abs(self.der(self.x0) - first_member + second_member)
        return abs_error

    def six_points_error(
            self,
            step: float
    ) -> float:
        first_member = 3 * (self.func(self.x0 + step) - self.func(self.x0 - step)) / (4 * step)
        second_member = 3 * (self.func(self.x0 + 2 * step) - self.func(self.x0 - 2 * step)) / (20 * step)
        third_member = (self.func(self.x0 + 3 * step) - self.func(self.x0 - 3 * step)) / (60 * step)
        abs_error = np.abs(self.der(self.x0) - first_member + second_member - third_member)
        return abs_error

