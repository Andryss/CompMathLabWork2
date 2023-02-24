import math
from typing import Callable


# noinspection DuplicatedCode
class Function:
    string: str = ""
    func: Callable[[float], float] = lambda x: 0

    def __init__(self, s, f):
        self.string = s
        self.func = f

    def at(self, x: float) -> float:
        return self.func(x)

    def derivative_at(self, x: float, precision: float = 1e-5) -> float:
        step = precision / 2

        left = self.at(x - step)
        center = self.at(x)
        right = self.at(x + step)

        left_d = (center - left) / step
        center_d = (right - left) / precision
        right_d = (right - center) / step

        return 1/3 * left_d + 1/3 * center_d + 1/3 * right_d

    def double_derivative_at(self, x: float, precision: float = 1e-5) -> float:
        step = precision / 2

        left_d = self.derivative_at(x - step)
        center_d = self.derivative_at(x)
        right_d = self.derivative_at(x + step)

        left_dd = (center_d - left_d) / step
        center_dd = (right_d - left_d) / precision
        right_dd = (right_d - center_d) / step

        return 1/3 * left_dd + 1/3 * center_dd + 1/3 * right_dd

    def has_one_root_on_interval(self, left: float, right: float, number_of_intervals_to_split: int = 1000) -> bool:
        assert right > left, "Wrong interval"
        if (right - left) / number_of_intervals_to_split > 0.5:
            raise Exception("Given interval is too big")

        left_val = self.at(left)
        right_val = self.at(right)

        if left_val * right_val > 0:
            return False

        idx = left
        step = (right - left) / number_of_intervals_to_split
        is_pos_derivative = self.derivative_at(idx) > 0
        while idx < right:
            if (self.derivative_at(idx) > 0) ^ is_pos_derivative:
                return False
            idx += step
        return True


def _get_polynomial_function() -> Function:
    return Function(
        "-0.38 * x^3 - 3.42 * x^2 + 2.51 * x + 7.75",
        lambda x: -0.38 * x**3 - 3.42 * x**2 + 2.51 * x + 8.75
    )


def _get_trigonometric_function() -> Function:
    return Function(
        "cos(x^2)",
        lambda x: math.cos(x ** 2)
    )


def get_all_functions() -> list[Function]:
    return [
        _get_polynomial_function(),
        _get_trigonometric_function()
    ]
