# -*- coding: utf-8 -*-
"""
power_series_example
====================
Script to compare performance of the power series approximation in Python and
in Fortran.

Variable names below match the notation in:

    http://www.christhoung.com/2017/12/03/f2py-power-series/

The example input-output table is from the *Tiny* model in

    Almon (2017) *The craft of economic modeling*,
    Third, enlarged edition, *Inforum*
    http://www.inforum.umd.edu/papers/TheCraft.html
"""

import numpy as np


# Define example variables ----------------------------------------------------
# Intermediate consumption
Z = np.array([
    [ 20.0,   1.0,   0.0, 100.0,   5.0,   0.0,   2.0,   0.0],
    [  4.0,   3.0,  20.0,  15.0,   2.0,   1.0,   2.0,   0.0],
    [  6.0,   4.0,  10.0,  40.0,  20.0,  10.0,  25.0,   0.0],
    [ 20.0,  10.0,   4.0,  60.0,  25.0,  18.0,  20.0,   0.0],
    [  2.0,   1.0,   1.0,  10.0,   2.0,   3.0,   6.0,   0.0],
    [  2.0,   1.0,   5.0,  17.0,   3.0,   2.0,   5.0,   0.0],
    [  6.0,   3.0,   8.0,  45.0,  20.0,   5.0,  20.0,   0.0],
    [  0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0]])

# Value added
v = np.array([[104.0], [ 27.0],
    [157.0],  # NB Value from Page 308, not Page 298
    [500.0], [324.0], [159.0], [587.0], [150.0]])

# Final demand
f = np.array([[ 36.0], [  3.0], [ 90.0], [630.0], [376.0], [163.0], [560.0], [150.0]])

# Gross output
q = np.array([[164.0], [ 50.0], [205.0], [787.0], [401.0], [198.0], [667.0], [150.0]])

# Direct requirements matrix
A = Z / q.T

# Check accounting ------------------------------------------------------------
# Row sums: q = Zi + f
assert np.allclose(q.flatten(), Z.sum(axis=1) + f.flatten())

# Column sums: q' = i'Z + v'
assert np.allclose(q.flatten(), Z.sum(axis=0) + v.flatten())


# Define power method functions -----------------------------------------------
def power_method_python(f, A, *, max_iter=100, tol=1e-06):
    """Python implementation of the power method.

    Parameters
    ----------
    f : (m x 1) vector
        Final demand
    A : (m x m) matrix
        Input-output matrix

    max_iter : integer
        Maximum number of iterations to run for
    tol : float
        Threshold for convergence

    Returns
    -------
    As a 3-tuple:
     - q : (m x 1) vector
           Gross output
     - converged : bool
           `True` if procedure converged; `False` otherwise
     - iterations : integer
           Number of iterations run for
    """
    q = f.copy()
    term = f.copy()

    converged = False

    for iterations in range(1, max_iter + 1):
        term = A @ term
        q += term

        if sum(term * term) < tol:
            converged = True
            break

    return q, converged, iterations


def power_method_fortran(f, A, *, max_iter=100, tol=1e-06):
    """Fortran implementation of the power method.

    Parameters
    ----------
    f : (m x 1) vector
        Final demand
    A : (m x m) matrix
        Input-output matrix

    max_iter : integer
        Maximum number of iterations to run for
    tol : float
        Threshold for convergence

    Returns
    -------
    As a 3-tuple:
     - q : (m x 1) vector
           Gross output
     - converged : bool
           `True` if procedure converged; `False` otherwise
     - iterations : integer
           Number of iterations run for
    """
    from leontief import power_series_approximation

    q, converged, iterations = power_series_approximation(f, A, max_iter, tol)

    return q.reshape(f.shape), bool(converged), iterations


if __name__ == '__main__':
    import timeit

    # Check results -----------------------------------------------------------
    # Leontief inverse
    L = np.linalg.inv(np.eye(A.shape[0]) - A)
    assert np.allclose(q, L @ f)

    # Power method (Python)
    results_python = power_method_python(f, A)
    assert results_python[1]  # Check for convergence
    assert np.allclose(q, results_python[0])

    # Power method (Fortran)
    results_fortran = power_method_fortran(f, A)
    assert results_fortran[1]  # Check for convergence
    assert np.allclose(q, results_fortran[0])

    # Test performance --------------------------------------------------------
    n = 10000
    time_leontief = timeit.timeit('np.linalg.inv(np.eye(A.shape[0]) - A) @ f', number=n, globals=globals())
    time_python = timeit.timeit('power_method_python(f, A)', number=n, globals=globals())
    time_fortran = timeit.timeit('power_method_fortran(f, A)', number=n, globals=globals())

    print('''Times (speed gain relative to Python power method):
 - Python (matrix inversion): {:.2f}s ({:.2f}x)
 - Python (power method): {:.2f}s ({:.2f}x)
 - Fortran (power method): {:.2f}s ({:.2f}x)'''.format(time_leontief, time_python / time_leontief,
                                                       time_python, time_python / time_python,
                                                       time_fortran, time_python / time_fortran))
