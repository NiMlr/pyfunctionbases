'''
ToDo:
'''

import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from pyfunctionbases.expansions import *
import numpy as np
from numpy.testing import assert_array_almost_equal
import time

DECIMAL = 7


def get_chebyshev_poly(x):
    p = np.ndarray((x.shape[0], 7))
    p[:, 0] = 1.
    p[:, 1] = x
    p[:, 2] = 2.*x*x-1.
    p[:, 3] = 4.*x**3-3.*x
    p[:, 4] = 8.*x**4-8.*x**2+1.
    p[:, 5] = 16.*x**5-20.*x**3+5.*x
    p[:, 6] = 32.*x**6-48.*x**4+18.*x*x-1.
    return p


def get_legendre_ratio(x):
    p = np.ndarray((x.shape[0], 5))
    p[:, 0] = 1.
    p[:, 1] = (x-1.)/(x+1.)
    p[:, 2] = (x*x-4.*x+1.)/(x+1.)**2
    p[:, 3] = (x**3-9.*x*x+9.*x-1.)/(x+1.)**3
    p[:, 4] = (x**4-16.*x**3+36.*x*x-16.*x+1.)/(x+1.)**4
    return p


def get_legendre_poly(x):
    p = np.ndarray((x.shape[0], 7))
    p[:, 0] = 1.
    p[:, 1] = x
    p[:, 2] = 0.5*(3.*x*x-1.)
    p[:, 3] = 0.5*(5.*x*x-3.)*x
    p[:, 4] = 1./8.*(35.*x**4-30.*x*x+3.)
    p[:, 5] = 1./8.*(63.*x**5-70.*x**3+15.*x)
    p[:, 6] = 1./16.*(231.*x**6-315.*x**4+105.*x*x-5.)
    return p


def get_standard_poly(x):
    p = np.ndarray((x.shape[0], 7))
    p[:, 0] = 1.
    p[:, 1] = x
    p[:, 2] = x*x
    p[:, 3] = x**3
    p[:, 4] = x**4
    p[:, 5] = x**5
    p[:, 6] = x**6
    return p


def get_handcomputed_function_tensor(x, func, degree):
    """x must be of shape (3,)."""
    outtensor = np.zeros((degree+1,)*3)

    outtensor[:, 0, 0] = func(x[np.newaxis, 0])
    outtensor[0, :, 0] = func(x[np.newaxis, 1])
    outtensor[0, 0, :] = func(x[np.newaxis, 2])

    for i in range(degree+1):
        outtensor[:, i, 0] = outtensor[:, 0, 0]*outtensor[0, i, 0]

    for i in range(degree+1):
        outtensor[:, :, i] = outtensor[:, :, 0]*outtensor[0, 0, i]

    return outtensor


funcs = [(get_standard_poly, 6, 'standard_poly'),
         (get_legendre_poly, 6, 'legendre_poly'),
         (get_legendre_ratio, 4, 'legendre_rational'),
         (get_chebyshev_poly, 6, 'chebyshev_poly')]


def test_RecursiveExpansionNode1():
    """Testing the one-dimensional expansion."""
    for functup in funcs:
        func = functup[0]
        degree = functup[1]
        name = functup[2]
        data = np.random.rand(1, 1)

        expn = RecursiveExpansion(degree, recf=name)
        nodeexp = expn.execute(data)
        assert_array_almost_equal(nodeexp,
                                  func(data[:, 0]), DECIMAL-3)
        print('Single dim '+name + ' equal')


def test_RecursiveExpansionNode2():
    """Testing the tensor-base."""
    data = 1e-6+np.random.randn(4, 3)
    for functup in funcs:
        func = functup[0]
        degree = functup[1]
        name = functup[2]
        recexpn = RecursiveExpansion(degree, recf=name)
        resrec = recexpn.execute(data)

        reshand = np.array([get_handcomputed_function_tensor(data[i, :], func, degree)
                            for i in range(data.shape[0])])

        if name == 'legendre_rational':
            prec = 1
        else:
            prec = 0
        assert_array_almost_equal(resrec, reshand, DECIMAL-3-prec)
        print('Multi dim ' + name + ' equal')


def test_Runtime():
    data = np.random.rand(100, 2)
    degree = 120
    REnode = RecursiveExpansion(degree, recf='legendre_poly')

    REstart = time.time()
    REshape = REnode.execute(data).shape
    REtime = time.time()-REstart

    print('Computation took: % f for % d polynomials' %
          (REtime, (degree+1)**data.shape[1]))
    print('Shapes: ', REshape)


if __name__ == "__main__":
    test_RecursiveExpansionNode1()
    test_RecursiveExpansionNode2()
    test_Runtime()
