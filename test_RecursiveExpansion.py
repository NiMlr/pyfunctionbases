'''
ToDo:
  - tests
'''
from mdp.nodes import PolynomialExpansionNode
from RecusiveExpansionNodeNils import *
from mdp.test._tools import *
import numpy as np


def get_tschebyschow_poly(x):
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
         (get_tschebyschow_poly, 6, 'tschebyschow_poly')]


def test_RecursiveExpansionNode1():
    """Testing the one-dimensional expansion."""
    for functup in funcs:
        func = functup[0]
        degree = functup[1]
        name = functup[2]
        data = np.random.rand(1, 1)

        expn = RecursiveExpansionNodeNils(degree, recf=name, transform=False)
        nodeexp = expn.execute(data)
        assert_array_almost_equal(nodeexp,
                                  func(data[:, 0]), decimal-3)
        print('Single dim '+name + ' equal')


def test_RecursiveExpansionNode2():
    """Testing the tensor-base."""
    data = 1e-6+np.random.randn(4, 3)
    for functup in funcs:
        func = functup[0]
        degree = functup[1]
        name = functup[2]
        recexpn = RecursiveExpansionNodeNils(degree, recf=name,
                                             transform=False)
        resrec = recexpn.execute(data)

        reshand = np.array([get_handcomputed_function_tensor(data[i, :], func, degree)
                            for i in range(data.shape[0])])

        assert_array_almost_equal(resrec, reshand, decimal-3)
        print('Multi dim ' + name + ' equal')


def test_Runtime():
    data = np.random.rand(100, 2)
    degree = 120
    steffannode = LegendrePolyExpansionNode2(degree)
    nilsnode = RecursiveExpansionNodeNils(degree, recf='legendre_poly',
                                          transform=False)

    start = time.time()
    nilsshape = nilsnode.execute(data).shape
    nilstime = time.time()-start

    start = time.time()
    stefanshape = steffannode.execute(data).shape
    stefantime = time.time()-start

    start = time.time()
    L = Legendre(data, degree)
    orthnetshape = L.tensor.shape
    orthnettime = time.time()-start

    print('Nils\' took: % f for % d polynomials' %
          (nilstime, (degree+1)**data.shape[1]))
    print('Stefans took: %f for %d polynomials' % (stefantime, stefanshape[1]))
    print('Orthnet took: %f for %d polynomials' %
          (orthnettime, orthnetshape[1]))
    print('Shapes: ', nilsshape, stefanshape, orthnetshape)


if __name__ == "__main__":
    test_RecursiveExpansionNode1()
    test_RecursiveExpansionNode2()
    test_Runtime()
