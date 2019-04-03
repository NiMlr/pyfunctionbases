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


def recf_tschebyschow_poly(basetensor, ind, x):
    "Implementation of the recursion formula for Tschebyschow polynomials."

    return 2. * x[:, ind.current_var] * basetensor[ind.all+ind.getPreceding(1)] \
        - basetensor[ind.all+ind.getPreceding(2)]


recfs = {'standard_poly': (2, 2, 1, init_standard_poly, recf_standard_poly,
                           -float('Inf'), float('Inf')),
         'legendre_poly': (2, 3, 1, init_standard_poly, recf_legendre_poly,
                           -1, 1),
         'legendre_rational': (2, 3, 0, init_legendre_rational,
                               recf_legendre_rational, 0, float('Inf')),
         'tschebyschow_poly': (2, 3, 1, init_standard_poly,
                               recf_tschebyschow_poly, -1, 1)}


class RecursiveExpansion(object):
    """Recursively computable (orthogonal) expansions."""

    def __init__(self, degree, recf='standard_poly', transform=True,
                 input_dim=None, dtype=None):
        """Initialize a RecursiveExpansionNode."""

        self.transform = transform
        self.degree = degree
        # if in dictionary
        if recf in recfs:
            # where the recursion starts
            self.rec_start = recfs[recf][0]
            # number of elements preceding to consider in the recursive step
            # includes the current element; does not a special first
            # such as x1,...,xn
            self.reach = recfs[recf][1]
            self.first = recfs[recf][2]
            # intialises the elements not based on recursion formula
            self.r_init = recfs[recf][3]
            # the recursion function
            self.recf = recfs[recf][4]
            # interval on which data must be
            self.upper = recfs[recf][5]
            self.lower = recfs[recf][6]
        # if supplied by user
        else:
            self.rec_start = recf[0]
            self.reach = recf[1]
            self.first = recf[2]
            self.r_init = recf[3]
            self.recf = recf[4]
            self.upper = recf[5]
            self.lower = recf[6]

    def expanded_dim(self, num_vars):
        """Return the size of a vector of dimension 'dim' after
        an expansion of degree 'self._degree'."""
        return (self.degree+1)**num_vars

    def execute(self, x):
        """Expansion of the data."""
        if self.transform:
            self._transform_orthogonal_interval(x)

        deg = self.degree
        num_vars = x.shape[1]
        num_samples = x.shape[0]
        # preset memory
        basetensor = np.zeros((num_samples,)+(deg+1,)*num_vars)
        # initialize index helper
        ind = BasisTensorIndicator(num_vars, self.reach, first=self.first)

        # set elements not in the recursion
        self.r_init(basetensor, ind, x)

        for cur_var in range(num_vars):
            # preset index for current variable
            ind.zeroAllBut(cur_var, self.rec_start)
            # single variable recursion
            while ind.getN() <= deg:
                # recursion step
                basetensor[ind.all+ind.getCurrent()] \
                    = self.recf(basetensor, ind, x)
                # next step
                ind.incrementN()

        # inplace tensorproduct
        np.einsum(ind.einsum_notation, *ind.getElementaries(basetensor),
                  out=basetensor, optimize='optimal')
        return basetensor

    def _transform_orthogonal_interval(self, x):
        """Transform the data onto the cube on which the
        the functions are orthogonal.

        If the cube is infinite the data is translated by the shortest
        length vector possible.

        If the cube is finite the data is scaled around the mean if neccessary.
        Then the datamean is moved onto the cube mean by a translation."""
        if self.lower is None or self.upper is None:
            def f(y): return y
            self.interval_transformation = f

        if self.lower == -float('Inf') and self.upper == float('Inf'):
            def f(y): return y
            self.interval_transformation = f
        elif self.lower == -float('Inf'):
            self.diff = np.amax(x, axis=0)-self.upper
            self.diff = self.diff.clip(min=0)
            x -= self.diff

            def f(y):
                y -= self.diff
                return y
            self.interval_transformation = f
        elif self.upper == float('Inf'):
            self.diff = np.amin(x, axis=0)-self.lower
            self.diff = self.diff.clip(max=0)
            x -= self.diff

            def f(y):
                y -= self.diff
                return y
            self.interval_transformation = f
        else:
            mean = self.lower+(self.upper-self.lower)/2.0
            dev = (self.upper-self.lower)/2

            datamean = np.mean(x, 0)
            x -= datamean
            datamaxdev = np.amax(np.abs(x))

            def f(y):
                y -= datamean
                y += mean
                return y

            if np.abs(datamaxdev) > 0:
                if dev/datamaxdev < 1:
                    x *= dev/datamaxdev

                    def f(y):
                        y -= datamean
                        y *= dev/datamaxdev
                        y += mean
                        return y

            x += mean

            self.interval_transformation = f


class BasisTensorIndicator(object):
    """Helps manage the indices of an up to 52 dimensional multi-dim array."""

    letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

    def __init__(self, num_vars, reach, first=1):
        # create index array of the last elements
        self.indices = np.zeros((reach+1, num_vars), dtype=int)
        self.num_vars = num_vars
        # how many last elements are to consider
        self.reach = reach
        self.current_var = 0
        # a special element used in every recursive step (e.g. x)
        self.first = first
        # replaces ":"" when tuple indexing numpy array
        self.all = (slice(None),)

        # generate the einstein summation convention string
        # for numpys einsum function
        self.einsum_notation = ''

        for i in range(num_vars):
            self.einsum_notation += self.letters[0]+self.letters[i+1]+','

        self.einsum_notation = self.einsum_notation[:-1] + \
            '->'+self.letters[:num_vars+1]

    def zeroAllBut(self, current_var, value=0):
        """Zeros every index besides the column of the current variable.
        That column contains decreasing indices starting with the current
        iteration step of the recursion."""
        self.indices[:, self.current_var] = 0
        self.current_var = current_var
        self.indices[0, current_var] = self.first
        for i in range(self.reach):
            self.indices[i+1, self.current_var] = value-i

    def incrementN(self, value=1):
        self.indices[1:, self.current_var] += value

    def getCurrent(self):
        return tuple(self.indices[1, :])

    def getPreceding(self, m):
        return tuple(self.indices[1+m, :])

    def getFirst(self):
        return tuple(self.indices[0, :])

    def getN(self):
        return self.indices[1, self.current_var]

    def getElementaries(self, basetensor):
        indexlist = [0]*self.num_vars
        elementaries = []
        for i in range(self.num_vars):
            indexlist[i] = slice(None)
            elementaries.append(basetensor[self.all+tuple(indexlist)])
            indexlist[i] = 0
        return elementaries
