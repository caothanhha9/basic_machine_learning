from sympy import Symbol
from sympy.functions import exp, sqrt
from math import pi


def test_standard_normal():
    x = Symbol('x')
    p = exp(-x**2/2) / sqrt(2 * pi)
    x0 = -1
    p_x_0 = p.subs(x, x0)
    print(float(p_x_0))

if __name__ == '__main__':
    test_standard_normal()
