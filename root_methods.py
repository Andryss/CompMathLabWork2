import pandas as pd
from functions import Function


class RootFindMethod:
    string: str = ""

    def evaluate_root(self, func: Function, left: float, right: float, precision: float = 1e-4) -> pd.DataFrame:
        raise Exception("Method isn't overridden")

    def extract_answer(self, result: pd.DataFrame) -> float:
        raise Exception("Method isn't overridden")

    def __str__(self):
        return self.string


class HalfDivisionMethod(RootFindMethod):
    string: str = "half division method"
    _half_division_method_table_cols = ["a", "b", "x", "f(a)", "f(b)", "f(x)", "|a - b|"]

    def evaluate_root(self, func: Function, left: float, right: float, precision: float = 1e-4) -> pd.DataFrame:
        table: list[list] = list()
        while True:
            line = [left, right]

            x = left + (right - left) / 2
            line.append(x)

            at_left = func.at(left)
            line.append(at_left)

            at_right = func.at(right)
            line.append(at_right)

            at_x = func.at(x)
            line.append(at_x)

            interval = right - left
            line.append(interval)

            table.append(line)

            if at_left * at_x < 0:
                right = x
            else:
                left = x

            if interval < precision or abs(at_x) < precision:
                break

        return pd.DataFrame(data=table, columns=self._half_division_method_table_cols)

    def extract_answer(self, result: pd.DataFrame) -> float:
        return result.values[-1][2]


class ChordMethod(RootFindMethod):
    string: str = "chord method"
    _chord_method_table_cols = ["a", "b", "x", "f(a)", "f(b)", "f(x)", "|x_(n+1) - x_n|"]

    def evaluate_root(self, func: Function, left: float, right: float, precision: float = 1e-4) -> pd.DataFrame:
        table: list[list] = list()
        last_x = left
        while True:
            line = [left, right]

            at_left = func.at(left)
            at_right = func.at(right)

            x = (left * at_right - right * at_left) / (at_right - at_left)
            line.append(x)

            line.append(at_left)
            line.append(at_right)

            at_x = func.at(x)
            line.append(at_x)

            interval = right - left
            change = abs(last_x - x)
            line.append(change)

            table.append(line)

            last_x = x

            if at_left * at_x < 0:
                right = x
            else:
                left = x

            if interval < precision or change < precision or abs(at_x) < precision:
                break

        return pd.DataFrame(data=table, columns=self._chord_method_table_cols)

    def extract_answer(self, result: pd.DataFrame) -> float:
        return result.values[-1][2]


class NewtonMethod(RootFindMethod):
    string: str = "newton method"
    _newton_method_table_cols = ["x_k", "f(x_k)", "f'(x_k)", "x_(k+1)", "|x_(k+1) - x_k|"]

    def evaluate_root(self, func: Function, left: float, right: float, precision: float = 1e-4) -> pd.DataFrame:
        table: list[list] = list()
        x = left if (func.at(left) * func.double_derivative_at(left) > 0) else right
        while True:
            line = [x]

            at_x = func.at(x)
            line.append(at_x)

            derivative_at_x = func.derivative_at(x)
            line.append(derivative_at_x)

            step = at_x / derivative_at_x
            next_x = x - step
            line.append(next_x)

            change = abs(next_x - x)
            line.append(change)

            table.append(line)

            x = next_x

            if change < precision or abs(step) < precision or abs(at_x) < precision:
                break

        return pd.DataFrame(data=table, columns=self._newton_method_table_cols)

    def extract_answer(self, result: pd.DataFrame) -> float:
        return result.values[-1][3]


class SecantMethod(RootFindMethod):
    string: str = "secant method"
    _secant_method_table_cols = ["x_(k-1)", "x_k", "x_(k+1)", "f(x_(k+1))", "|x_(k+1) - x_k|"]

    def evaluate_root(self, func: Function, left: float, right: float,
                      precision: float = 1e-4, first_offset: float = 0.1) -> pd.DataFrame:
        table: list[list] = list()
        prev_x = left if (func.at(left) * func.double_derivative_at(left) > 0) else right
        x = (prev_x + first_offset) if (prev_x == left) else (prev_x - first_offset)
        while True:
            line = [prev_x, x]

            at_prev_x = func.at(prev_x)
            at_x = func.at(x)
            next_x = x - (x - prev_x) / (at_x - at_prev_x) * at_x
            line.append(next_x)

            at_next_x = func.at(next_x)
            line.append(at_next_x)

            change = abs(next_x - x)
            line.append(change)

            table.append(line)

            prev_x = x
            x = next_x

            if change < precision or abs(at_next_x) < precision:
                break

        return pd.DataFrame(data=table, columns=self._secant_method_table_cols)

    def extract_answer(self, result: pd.DataFrame) -> float:
        return result.values[-1][2]


class SimpleIterationMethod(RootFindMethod):
    string: str = "simple iteration method"
    _simple_iteration_method_table_cols = ["x_k", "x_(k+1)", "f(x_(k+1))", "|x_(k+1) - x_k|"]

    def evaluate_root(self, func: Function, left: float, right: float,
                      precision: float = 1e-4, number_of_steps: int = 10000) -> pd.DataFrame:
        table: list[list] = list()

        k = abs(func.derivative_at(left))
        step = (right - left) / number_of_steps
        idx = left + step
        while idx < right + step:
            k = max(k, abs(func.derivative_at(idx)))
            idx += step
        lambda_coefficient = - 1 / k

        transformed_func = Function(
            "x + lambda * f(x)",
            lambda x: x + lambda_coefficient * func.at(x)
        )

        stopped_x = self._try_iteration(func, transformed_func, table, left, right, precision)

        if stopped_x < left or stopped_x > right:
            table = list()
            transformed_func = Function(
                "x - lambda * f(x)",
                lambda x: x - lambda_coefficient * func.at(x)
            )
            stopped_x = self._try_iteration(func, transformed_func, table, left, right, precision)

            if stopped_x < left or stopped_x > right:
                raise Exception("Simple iteration method is annihilated (mission accomplished)")

        return pd.DataFrame(data=table, columns=self._simple_iteration_method_table_cols)

    @staticmethod
    def _try_iteration(func: Function, transformed_func: Function, table: list[list],
                       left: float, right: float, precision: float) -> float:
        x = left
        while left <= x <= right:
            line = [x]

            next_x = transformed_func.at(x)
            line.append(next_x)

            at_next_x = func.at(next_x)
            line.append(at_next_x)

            change = abs(next_x - x)
            line.append(change)

            table.append(line)

            x = next_x

            if change < precision:
                break
        return x

    def extract_answer(self, result: pd.DataFrame) -> float:
        return result.values[-1][1]


def get_all_methods() -> list[RootFindMethod]:
    return [
        HalfDivisionMethod(),
        ChordMethod(),
        NewtonMethod(),
        SecantMethod(),
        SimpleIterationMethod()
    ]
