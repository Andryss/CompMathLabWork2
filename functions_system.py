import math
from typing import Callable


# noinspection DuplicatedCode
class ManyArgumentFunction:
    string: str = ""
    argc: int = 0
    func: Callable[[list[float]], float] = lambda args: 0

    def __init__(self, s, a, f):
        self.string = s
        self.argc = a
        self.func = f

    def at(self, x: list[float]) -> float:
        assert self.argc == len(x), "Wrong amount of arguments"
        return self.func(x)

    def partial_derivative_at(self, x: list[float], to: int, precision: float = 1e-5) -> float:
        assert 0 <= to < len(x), "Argument is out of range"
        step = precision / 2

        tmp = x.copy()
        tmp[to] -= step
        left = self.at(tmp)
        center = self.at(x)
        tmp[to] += 2 * step
        right = self.at(tmp)

        left_d = (center - left) / step
        center_d = (right - left) / precision
        right_d = (right - center) / step

        return 1/3 * left_d + 1/3 * center_d + 1/3 * right_d

    def __str__(self):
        return "function: (" + self.string + ")"


class EquationSystem:
    image: str = ""
    funcs: list[ManyArgumentFunction] = None

    def __init__(self, im: str, fs: list[ManyArgumentFunction]):
        assert len(fs) > 0, "Not enough functions"
        args_amount = fs[0].argc
        assert all(map(lambda x: x.argc == args_amount, fs)), "All functions must have same amount of args"
        assert len(fs) == args_amount, "Amount of functions must be equal to amount of args"
        self.image = im
        self.funcs = fs

    def __str__(self):
        top = self.funcs[0]
        middle = self.funcs[1:-1]
        bottom = self.funcs[-1]
        system = "/ " + top.string
        for func in middle:
            system += "\n| " + func.string
        system += "\n\\ " + bottom.string
        return system


def _get_classes_system() -> EquationSystem:
    return EquationSystem(
        "system_plots/0.png",
        [
            ManyArgumentFunction(
                "x_0 = 0.3 - 0.1 * x_0^2 - 0.2 * x_1^2",
                2,
                lambda args: 0.3 - 0.1 * args[0]**2 - 0.2 * args[1]**2     # == args[0]
            ),
            ManyArgumentFunction(
                "x_1 = 0.7 - 0.2 * x_0^2 - 0.1 * x_0 * x_1",
                2,
                lambda args: 0.7 - 0.2 * args[0]**2 - 0.1 * args[0] * args[1]     # == args[1]
            )
        ]
    )


def _get_internet_system() -> EquationSystem:
    return EquationSystem(
        "system_plots/1.png",
        [
            ManyArgumentFunction(
                "x_0 = 1/3 * cos(x_1) + 1.3",
                2,
                lambda args: 1/3 * math.cos(args[1]) + 1.3     # == args[0]
            ),
            ManyArgumentFunction(
                "x_1 = sin(x_0 - 0.6) - 1.6",
                2,
                lambda args: math.sin(args[0] - 0.6) - 1.6     # == args[1]
            )
        ]
    )


def get_all_equation_systems() -> list[EquationSystem]:
    return [
        _get_classes_system(),
        _get_internet_system()
    ]
