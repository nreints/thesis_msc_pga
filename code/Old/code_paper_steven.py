import torch

def log(r):
    """
    Input rotor (8 numbers) returns bivector (=log of rotor) (6 numbers)
    (14 mul, 5 add, 1 div, 1 acos, 1 sqrt)
    """
    if r[0] == 1:
        return bivector(r[1], r[2], r[3], 0, 0, 0)
    a = 1 / (1 - r[0] * r[0])
    b = acos(r[0]) * sqrt(a)
    c = a * r[7] * (1 - r[0] * b)
    return bivector(
                    c * r[6] + b * r[1], \
                    c * r[5] + b * r[2], \
                    c * r[4] + b * r[3], \
                    b * r[4], \
                    b * r[5], \
                    b * r[6]
                )


def exp(b):
    """
    Input bivector (6 numbers) returns rotor (=exp of bivector) (8 numbers)
    (17 mul, 8 add, 2 div, 1 sincos, 1 sqrt)
    """
    l = b[3] * b[3] + b[4] * b[4] + b[5] * b[5]
    if l == 0:
        return rotor(1, b[0], b[1], b[2], 0, 0, 0, 0)

    m = b[0] * b[5] + b[1] * b[4] + b[2] * b[3]
    a = sqrt(l)
    c = cos(a)
    s = sin(a) / a
    t = m / l * (c - s)
    return rotor(
                    c,
                    s * b[0] + t * b[5],
                    s * b[1] + t * b[4],
                    s * b[2] + t * b[3],
                    s * b[3],
                    s * b[4],
                    s * b[5],
                    m * s
                )


def normalize(x):
    """
    Normalize an even element X on the basis [1,e01,e02,e03,e12,e31,e23,e0123]
    """
    a = 1 / (x[0] * x[0] + x[4] * x[4] + x[5] * x[5] + x[6] * x[6])**0.5
    b = (x[7] * x[0] - (x[1] * x[6] + x[2] * x[5] + x[3] * x[4])) * a * a * a
    return rotor(
                    a*x[0],
                    a*x[1]+b*x[6],
                    a*x[2]+b*x[5],
                    a*x[3]+b*x[4],
                    a*x[4],
                    a*x[5],
                    a*x[6],
                    a*x[7]-b*x[0]
                )


def square_root(r):
    """
    Square root of a rotor R
    """
    return normalize(1 + r)
