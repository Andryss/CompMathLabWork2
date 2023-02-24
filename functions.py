from typing import Callable


class Function:
    func: Callable[[float], float] = None

    def __init__(self, f):
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


def get_polynomial_function() -> Function:
    return Function(lambda x: -0.38 * x**3 - 3.42 * x**2 + 2.51 * x + 8.75)


f1 = get_polynomial_function()
print(f1.double_derivative_at(-3))
