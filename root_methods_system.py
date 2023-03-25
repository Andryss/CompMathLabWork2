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
        SimpleIterationSystemMethod.check_usability(system, intervals)

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

        iterations = 0
        while iterations < max_iterations:
            line = []

            new_point = point.copy()
            is_all_less_then_precision = True
            for i in range(variables_count):
                new_point[i] = system.funcs[i].at(point)
                line.append(new_point[i])
                change = abs(point[i] - new_point[i])
                line.append(change)
                if change > precision:
                    is_all_less_then_precision = False

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
    def check_usability(system: EquationSystem, intervals: list[list[float]], number_of_steps: int = 1000):
        if len(system.funcs) != 2:
            raise Exception("Method isn't prepared for that type of systems")

        step_0 = (intervals[0][1] - intervals[0][0]) / number_of_steps
        step_1 = (intervals[1][1] - intervals[1][0]) / number_of_steps

        if step_0 > 0.1 or step_1 > 0.1:
            raise Exception("Given interval is too big to check it (reduce the intervals)")

        start_0 = intervals[0][0]
        max_part_der_0, max_part_der_1 = 0, 0

        for i in range(number_of_steps + 1):
            start_1 = intervals[1][0]
            for j in range(number_of_steps + 1):
                part_der_0at0 = abs(system.funcs[0].partial_derivative_at([start_0, start_1], 0))
                part_der_0at1 = abs(system.funcs[0].partial_derivative_at([start_0, start_1], 1))
                max_part_der_0 = max(max_part_der_0, part_der_0at0 + part_der_0at1)

                part_der_1at0 = abs(system.funcs[1].partial_derivative_at([start_0, start_1], 0))
                part_der_1at1 = abs(system.funcs[1].partial_derivative_at([start_0, start_1], 1))
                max_part_der_1 = max(max_part_der_1, part_der_1at0 + part_der_1at1)

                if part_der_0at0 + part_der_0at1 > 1:
                    raise Exception(f"Cannot use this method: first partial derivative more than 1 "
                                    f"(equal to {max_part_der_0} at point ({start_0},{start_1}))")

                if part_der_1at0 + part_der_1at1 > 1:
                    raise Exception(f"Cannot use this method: second partial derivative more than 1 "
                                    f"(equal to {max_part_der_1} at point ({start_0},{start_1}))")

                start_1 += step_1
            start_0 += step_0

        # print(max_part_der_0, max_part_der_1)


def get_all_system_methods() -> list[SystemRootFindMethod]:
    return [
        SimpleIterationSystemMethod()
    ]
