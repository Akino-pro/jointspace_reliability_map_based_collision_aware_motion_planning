import numpy as np

from poly_funcs import (
    Poly
)


class LinearInequality:
    def __init__(self, p):
        self.p = p

    def get_constraint(self, x):
        return [self.p.A@x <= self.p.b]

    def get_perspective_constraint(self, x, phi):
        return [self.p.A@x <= self.p.b*phi]


class ConstrainedLinearEquality:
    def __init__(self, p, c, d):
        self.p = p
        self.c = c
        self.d = d

    def get_constraint(self, x):
        return [self.p.A@x <= self.b, self.c@x == d]

    def get_perspective_constraint(self, x, phi):
        return [self.p.A@x <= self.b*phi, self.c@x == d*phi]


class Point:
    def __init__(self, b):
        self.b = b

    def get_constraint(self, x):
        return [x == self.b]

    def get_perspective_constraint(self, x, phi):
        return [x == self.b*phi]
