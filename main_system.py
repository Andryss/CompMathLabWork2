import sys
from PIL import Image

from root_methods_system import *


def choose_system() -> EquationSystem:
    systems: list[EquationSystem] = get_all_equation_systems()
    print("\nChoose the system you want to explore:")
    for i, system in enumerate(systems):
        print(str(i) + "\n" + system.__str__())
    line = input("(enter the number) ").strip()
    try:
        idx = int(line)
        if idx < 0 or idx >= len(systems):
            raise Exception("not such system")
        return systems[idx]
    except Exception as e:
        raise Exception("can't choose the function: " + e.__str__())


def show_plot(system: EquationSystem):
    try:
        image = Image.open(system.image)
        image.show(system.__str__())
        print("\nI drawn a plot for you, look at It before continue.")
    except Exception as e:
        raise Exception("can't show the plot: " + e.__str__())



def read_intervals(n: int) -> list[list[float]]:
    intervals = []
    print("\nEnter the interval boundaries for")
    try:
        for i in range(n):
            line = input(f"variable {i}: ").strip()
            interval = [float(x) for x in line.split()]
            if len(interval) != 2 or interval[0] >= interval[1]:
                raise Exception("not an interval")
            intervals.append(interval)
        return intervals
    except Exception as e:
        raise Exception("can't read the intervals: " + e.__str__())


def read_start(intervals: list[list[float]]) -> list[float]:
    start = []
    print("\nEnter the start value for")
    try:
        for i, interval in enumerate(intervals):
            line = input(f"variable {i} in range [{interval[0]}, {interval[1]}]: ").strip()
            value = float(line)
            if value < interval[0] or value > interval[1]:
                raise Exception("value out of the range")
            start.append(value)
        return start
    except Exception as e:
        raise Exception("can't read the start values: " + e.__str__())


def read_precision() -> float:
    try:
        line = input("\nEnter the precision:\n").strip()
        precision = float(line)
        if precision <= 0:
            raise Exception("precision must be positive")
        return precision
    except Exception as e:
        raise Exception("can't read the precision: " + e.__str__())


def choose_method() -> SystemRootFindMethod:
    methods: list[SystemRootFindMethod] = get_all_system_methods()
    print("\nChoose the method you want to use:")
    for i, method in enumerate(methods):
        print(str(i) + "\t" + method.__str__())
    line = input("(enter the number) ").strip()
    try:
        idx = int(line)
        if idx < 0 or idx >= len(methods):
            raise Exception("not such method")
        return methods[idx]
    except Exception as e:
        raise Exception("can't choose the method: " + e.__str__())


def print_result(result: pd.DataFrame, method: SystemRootFindMethod, system: EquationSystem):
    print("\nHere is the computation result:")

    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    print(result)

    answer = method.extract_answer(result)
    print("\nEnd the final answer is:")
    for i, ans in enumerate(answer):
        print(f"x_{i} = {ans}")

    print(f"Number of iterations: {len(result.values)}")

    print(f"Precisions:")
    for i, func in enumerate(system.funcs):
        print(f"{i}: {answer[i] - func.at(answer)}")


def run():
    try:
        system: EquationSystem = choose_system()
        show_plot(system)

        intervals: list[list[float]] = read_intervals(len(system.funcs))
        start: list[float] = read_start(intervals)
        precision: float = read_precision()
        method: SystemRootFindMethod = choose_method()

        result: pd.DataFrame = method.evaluate_root(system, intervals, start, precision)
        print_result(result, method, system)
        # show_plot(function, left, right, result=result, method=method)
    except Exception as e:
        print(e, file=sys.stderr)


if __name__ == '__main__':
    run()
