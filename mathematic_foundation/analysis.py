# -*- coding: utf-8 -*-
import numpy as np
from sympy import *
from sympy.functions import exp


def test_solve_equation():
    # y = f(x) = ax + b
    # Đồ thị hàm số đi qua điểm (0, 900) => 900 = a * 0 + b
    # Đồ thị hàm số đi qua điểm (100, 800) => 800 = a * 100 + b
    # Để tìm a,b ta hình thành 2 ma trận A, B sao cho: A * [a, b] = B
    matrix_a = [[0, 1], [100, 1]]
    matrix_b = [900, 800]
    solution = np.linalg.solve(matrix_a, matrix_b)
    print('Kết quả:')
    print('y = f(x) = ' + str(solution[0]) + 'x' + ' + ' + str(solution[1]))


def test_differentiate():
    x = Symbol('x')
    a = Symbol('a')
    b = Symbol('b')
    c = Symbol('c')
    y = a * x**2 + b * x + c
    yprime = y.diff(x)
    print('Đạo hàm của hàm số y = ax2 + bx + c')
    print(yprime)


def test_differentiate_sigmoid():
    x = Symbol('x')
    y = 1 / (1 + exp(-1 * x))
    yprime = y.diff(x)
    print('Đạo hàm của hàm số y = sigmoid(x)')
    print(yprime)


def test_differentiate_tanh():
    x = Symbol('x')
    y = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    yprime = y.diff(x)
    print('Đạo hàm của hàm số y = tanh(x)')
    print(yprime)


def test_differentiate_multivariate():
    x = Symbol('x')
    y = Symbol('y')
    t = Symbol('t')
    z = 3 * x**2 + 2 * x * y - y**2
    print('Đạo hàm riêng z theo x')
    zprimex = z.diff(x)
    zprimext = zprimex.subs(x, 2*t + 1).subs(y, t**2)
    print(zprimex)
    print('Đạo hàm riêng z theo y')
    zprimey = z.diff(y)
    zprimeyt = zprimey.subs(x, 2*t + 1).subs(y, t**2)
    print(zprimey)
    x = 2 * t + 1
    y = t**2
    xprimet = x.diff(t)
    yprimet = y.diff(t)
    print('Đạo hàm x theo t')
    print(xprimet)
    print('Đạo hàm y theo t')
    print(yprimet)
    print('Đạo hàm z theo t')
    zprimet = zprimext * xprimet + zprimeyt * yprimet
    print(zprimet)

if __name__ == "__main__":
    # test_solve_equation()
    # test_differentiate()
    # test_differentiate_sigmoid()
    # test_differentiate_tanh()
    test_differentiate_multivariate()
