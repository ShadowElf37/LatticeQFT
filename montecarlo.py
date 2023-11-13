from numpy import random
import numpy as np

def integrate(f, a, b, N=100):
    """Integrate f(x) over the domain (a,b), where x, a, and b can be arrays (must be the same dimension)"""
    try:
        dim = len(a)
    except TypeError:
        dim = 1
    box_size = (b - a)
    exp_f = 0
    for _ in range(N):
        exp_f += f(random.random(dim) * box_size + a)
    return box_size * exp_f / N

def integrate_with_error_bounds(f, a, b, N=100):
    """Same as other but return 1 sigma errors"""
    try:
        dim = len(a)
    except TypeError:
        dim = 1
    box_size = np.prod(b-a)
    exp_f = 0
    exp_f_squared = 0
    for _ in range(N):
        F = f(random.random(dim)*box_size+a)
        exp_f += F
        exp_f_squared += F**2
    exp_f /= N
    exp_f_squared /= N

    error = np.sqrt((exp_f_squared - exp_f**2)/N)
    return box_size*exp_f, box_size*error


if __name__ == "__main__":
    def f(x):
        return -x**2+1

    import matplotlib.pyplot as plot
    for n in (10, 30, 100, 300, 1000, 3000, 10000):
        res = integrate_with_error_bounds(f, -1, 1, n)
        plot.errorbar(n, res[0], yerr=res[1], color='blue', capsize=3, capthick=1, elinewidth=1, linestyle='none')
        plot.scatter(n, res[0], s=10, color='#00f')

    plot.axhline(4/3, color='green')
    plot.xscale('log')
    plot.show()