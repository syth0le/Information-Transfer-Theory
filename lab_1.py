import math

import matplotlib.pyplot as plt
import numpy as np

PI = np.pi
T = 2 * PI
W = T / T
A = T / 2
DOTS = np.arange(-PI, PI, 0.01)


def fn(a: float, x: float) -> float:
    return np.sinh(a * x)


def bn(a: float, n: float) -> float:
    return (2 * np.sinh(PI * a) / PI) * ((math.pow(-1, n + 1) * n) / (math.pow(a, 2) + math.pow(n, 2)))


def draw_graphic(f_t: np.array, F_t: np.array, N: int) -> None:
    plt.figure()
    plt.title(f'Классический гармонический синтез для N = {N}')
    plt.plot(DOTS, f_t, label='f(t)')
    plt.plot(DOTS, F_t, label='F(t)')
    plt.xlabel('Время t')
    plt.ylabel('f(t),F(t)')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    amount_elements = [3, 5, 10, 20]
    for elem in amount_elements:
        b = [bn(A, k) for k in np.arange(0, elem)]

        f = [fn(A, t) for t in DOTS]
        F = np.array([0 * t for t in DOTS - 1])
        for k in np.arange(1, elem):
            F = F + np.array([b[k] * np.sin(W * k * t) for t in DOTS])

        draw_graphic(f, F, elem)
