import sys
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from functions import *
from root_methods import *


def choose_function() -> Function:
    functions: list[Function] = get_all_functions()
    print("\nChoose the function you want to explore:")
    for i, function in enumerate(functions):
        print(str(i) + "\t" + function.__str__())
    line = input("(enter the number) ").strip()
    try:
        idx = int(line)
        if idx < 0 or idx >= len(functions):
            raise Exception("not such function")
        return functions[idx]
    except Exception as e:
        raise Exception("can't choose the function: " + e.__str__())


def read_interval() -> [float, float]:
    line = input("\nEnter the interval boundaries:\n").strip()
    interval: [float, float]
    try:
        interval = [float(x) for x in line.split()]
        if len(interval) != 2 or interval[1] < interval[0]:
            raise Exception("not an interval")
    except Exception as _:
        interval = read_interval_from_file(line)
    return interval


def read_interval_from_file(filename: str):
    frame: pd.DataFrame
    try:
        frame = pd.read_csv(filename, header=None)
        validate_interval_precision(frame)
    except Exception as e:
        raise Exception("file \"" + filename + "\" can't be opened: " + e.__str__())
    return frame.values


def validate_interval_precision(frame: pd.DataFrame):
    if len(frame) != 1 or len(frame[0]) != 2 or not isinstance(frame[0][0], np.float) \
            or not isinstance(frame[0][1], np.float) or frame[0][1] < frame[0][0]:
        raise Exception("must contains only two float numbers (forming interval)")


def read_precision() -> float:
    line = input("\nEnter the precision:\n").strip()
    precision: float
    try:
        precision = float(line)
        if precision <= 0:
            raise Exception("precision must be positive")
    except Exception as _:
        precision = read_precision_from_file(line)
    return precision


def read_precision_from_file(filename: str):
    frame: pd.DataFrame
    try:
        frame = pd.read_csv(filename, header=None)
        validate_file_precision(frame)
    except Exception as e:
        raise Exception("file \"" + filename + "\" can't be opened: " + e.__str__())
    return frame[0][0]


def validate_file_precision(frame: pd.DataFrame):
    if len(frame) != 1 or len(frame[0]) != 1 or not isinstance(frame[0][0], np.float) or frame[0][0] <= 0:
        raise Exception("must contains only one number (positive float)")


def choose_method() -> RootFindMethod:
    methods: list[RootFindMethod] = get_all_methods()
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


def print_result(result: pd.DataFrame, method: RootFindMethod):
    print("\nHere is the computation result:")
    print(result)
    print("\nEnd the final answer is: x =", method.extract_answer(result))


def show_plot(function: Function, left: float, right: float,
              result: pd.DataFrame = None, method: RootFindMethod = None):
    x = np.arange(left, right, (right - left) / 1000)
    y = np.array([function.at(val) for val in x])
    warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)
    plt.scatter(x, y)
    plt.plot([left, right], [0, 0], "black")
    if not (result is None and method is None):
        point, = plt.plot([method.extract_answer(result)], [0], "ro")
        plt.legend([point], [method.string])
    plt.title(f'({function.string}) at [{left}, {right}]')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


def run():
    try:
        function: Function = choose_function()
        [left, right] = read_interval()

        if not function.has_one_root_on_interval(left, right):
            show_plot(function, left, right)
            raise Exception("Sorry can't tell you anything about that interval of that function "
                            "(possibly there are 0 or more then 1 roots here)")

        precision: float = read_precision()
        method: RootFindMethod = choose_method()
        result: pd.DataFrame = method.evaluate_root(function, left, right, precision)
        print_result(result, method)
        show_plot(function, left, right, result, method)
    except Exception as e:
        print(e, file=sys.stderr)


if __name__ == '__main__':
    run()
