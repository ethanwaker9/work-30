import numpy as np


def cheb_nodes(m):
    k = np.arange(m)
    return np.cos(np.pi * (k + 0.5) / m), np.pi * (k + 0.5) / m


def cheb_coeffs(f, a, b, degree, oversample=4):
    m = max(oversample * (degree + 1), 256)
    x, th = cheb_nodes(m)
    y = np.asarray(f(0.5 * (b - a) * x + 0.5 * (b + a)), dtype=np.float64)
    j = np.arange(degree + 1)
    c = (2.0 / m) * (np.cos(np.outer(j, th)) @ y)
    c[0] /= 2.0
    return c


def cheb_eval(c, a, b, x):
    t = (2.0 * np.asarray(x, dtype=np.float64) - (a + b)) / (b - a)
    b0 = np.zeros_like(t)
    b1 = np.zeros_like(t)
    for j in range(len(c) - 1, 0, -1):
        b0, b1 = 2 * t * b0 - b1 + c[j], b0
    return t * b0 - b1 + c[0]


def sup_error(f, a, b, c, n_pts=20001, relative=False, log_grid=False):
    x = np.geomspace(a, b, n_pts) if log_grid else np.linspace(a, b, n_pts)
    fx = np.asarray(f(x), dtype=np.float64)
    e = np.abs(cheb_eval(c, a, b, x) - fx)
    if relative:
        e = e / np.maximum(np.abs(fx), 1e-300)
    return float(e.max())


def min_degree(f, a, b, tol, relative=False, d_max=1024, log_grid=False):
    c = cheb_coeffs(f, a, b, d_max)
    lo, hi = 0, d_max
    if sup_error(f, a, b, c, relative=relative, log_grid=log_grid) > tol:
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if sup_error(f, a, b, c[:mid + 1], relative=relative, log_grid=log_grid) <= tol:
            hi = mid
        else:
            lo = mid + 1
    return lo


def fit(f, a, b, tol, relative=False, d_max=1024, log_grid=False):
    d = min_degree(f, a, b, tol, relative, d_max, log_grid)
    if d is None:
        d = d_max
    return cheb_coeffs(f, a, b, d)


def poly_depth(degree):
    return int(np.ceil(np.log2(degree + 1))) if degree >= 1 else 0
