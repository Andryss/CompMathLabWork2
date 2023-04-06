import numpy as np
import pandas as pd

from functions_system import *


class SystemRootFindMethod:
    string: str = ""

    def evaluate_root(self, system: EquationSystem, intervals: list[list[float]], start: list[float],
                      precision: float = 1e-4) -> pd.DataFrame:
        raise Exception("Method isn't overridden")

    def extract_answer(self, result: pd.DataFrame) -> list[float]:
        raise Exception("Method isn't overridden")

    def __str__(self):
        return self.string


class SimpleIterationSystemMethod(SystemRootFindMethod):
    string: str = "simple iteration method"

    def evaluate_root(self, system: EquationSystem, intervals: list[list[float]], start: list[float],
                      precision: float = 1e-4, max_iterations: int = 10000) -> pd.DataFrame:

        table_cols = []
        variables_count = system.funcs[0].argc
        for i in range(variables_count):
            table_cols.append(f"x_{i}")
            table_cols.append(f"|x_{i}^k - x_{i}^(k-1)|")

        table = list()
        first_line = []
        for i in range(variables_count):
            first_line.append(start[i])
            first_line.append(pd.NA)
        table.append(first_line)

        point = start.copy()
        SimpleIterationSystemMethod.check_usability_at(system, point)

        iterations = 0
        while iterations < max_iterations:
            line = []

            new_point = point.copy()
            is_all_less_then_precision = True
            for i in range(variables_count):
                new_point[i] = system.funcs[i].at(point)
                if new_point[i] < intervals[i][0] or new_point[i] > intervals[i][1]:
                    raise Exception(f"Method iterated out of the searching area (new value of variable {i} is {new_point[i]} when interval is {intervals[i]})")
                line.append(new_point[i])
                change = abs(point[i] - new_point[i])
                line.append(change)
                if change > precision:
                    is_all_less_then_precision = False

            SimpleIterationSystemMethod.check_usability_at(system, point)
            table.append(line)

            if is_all_less_then_precision:
                break

            point = new_point
            iterations += 1

        if iterations == max_iterations:
            raise Exception("The maximum number of iterations reached. Method didn't complete")

        return pd.DataFrame(data=table, columns=table_cols)

    def extract_answer(self, result: pd.DataFrame) -> list[float]:
        answer = []
        last_row = result.values[-1]
        for i in range(0, len(last_row), 2):
            answer.append(last_row[i])
        return answer

    @staticmethod
    def check_usability_at(system: EquationSystem, point: list[float]):
        for func_num in range(len(system.funcs)):
            part_der_sum = 0
            for var_num in range(len(point)):
                part_der_sum += abs(system.funcs[func_num].partial_derivative_at(point, var_num))
            if part_der_sum > 1:
                raise Exception(f"Cannot use this method: partial derivative more than 1 "
                                f"(equal to {part_der_sum} at point {point})")


def get_all_system_methods() -> list[SystemRootFindMethod]:
    return [
        SimpleIterationSystemMethod()
    ]
