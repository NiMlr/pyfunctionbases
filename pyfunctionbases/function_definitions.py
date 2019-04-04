"""
ToDo:
  - add datatype support
  - maybe change var to dim
  - first should maybe be named special
  - make interval transformation look nicer
  - tests for interval transformation?
  - better documentation
  - implement more recursions
"""
import numpy as np


def recf_standard_poly(basetensor, ind, x):
    "Implementation of the recursion formula for standard polynomials."
    return basetensor[ind.all+ind.getPreceding(1)] * \
        basetensor[ind.all+ind.getFirst()]


def init_standard_poly(basetensor, ind, x):
    "Initialize the 1 and x1,...,xn before starting the recursion."
    ind.zeroAllBut(0, 0)
    basetensor[ind.all+ind.getCurrent()] = 1.
    for i in range(x.shape[1]):
        ind.zeroAllBut(i, 1)
        basetensor[ind.all + ind.getCurrent()] = x[:, i]


def recf_legendre_poly(basetensor, ind, x):
    "Implementation of the recursion formula for Legendre polynomials."
    n = ind.getN()
    return (2.*n-1.)/n*basetensor[ind.all+ind.getFirst()] \
        * basetensor[ind.all + ind.getPreceding(1)] \
        - (n-1.)/n*basetensor[ind.all + ind.getPreceding(2)]


def init_legendre_rational(basetensor, ind, x):
    "Initialize the 1 and x1,...,xn before starting the recursion."
    ind.zeroAllBut(0, 0)
    basetensor[ind.all+ind.getCurrent()] = 1.
    for i in range(x.shape[1]):
        ind.zeroAllBut(i, 1)
        basetensor[ind.all + ind.getCurrent()] = (x[:, i]-1.)/(x[:, i]+1.)


def recf_legendre_rational(basetensor, ind, x):
    "Implementation of the recursion formula for Legendre polynomials."
    n = ind.getN()
    xv = x[:, ind.current_var]
    Rnmin1 = basetensor[ind.all+ind.getPreceding(1)]
    Rnmin2 = basetensor[ind.all+ind.getPreceding(2)]
    return (2.*n-1.)/n*(xv-1.) / (xv+1.) * Rnmin1 - (n-1.) / n * Rnmin2


def recf_chebyshev_poly(basetensor, ind, x):
    "Implementation of the recursion formula for Chebyshev polynomials."

    return 2. * x[:, ind.current_var] * basetensor[ind.all+ind.getPreceding(1)] \
        - basetensor[ind.all+ind.getPreceding(2)]


recfs = {'standard_poly': (2, 2, 1, init_standard_poly, recf_standard_poly,
                           -float('Inf'), float('Inf')),
         'legendre_poly': (2, 3, 1, init_standard_poly, recf_legendre_poly,
                           -1, 1),
         'legendre_rational': (2, 3, 0, init_legendre_rational,
                               recf_legendre_rational, 0, float('Inf')),
         'chebyshev_poly': (2, 3, 1, init_standard_poly,
                            recf_chebyshev_poly, -1, 1)}
