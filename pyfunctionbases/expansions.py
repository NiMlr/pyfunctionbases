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
from .function_definitions import recfs


class RecursiveExpansion(object):
    """Recursively computable (orthogonal) expansions."""

    def __init__(self, degree, recf='standard_poly', input_dim=None,
                 dtype=None):
        """Initialize a RecursiveExpansionNode."""

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
            self.upper = recfs[recf][6]
            self.lower = recfs[recf][5]
        # if supplied by user
        else:
            self.rec_start = recf[0]
            self.reach = recf[1]
            self.first = recf[2]
            self.r_init = recf[3]
            self.recf = recf[4]
            self.upper = recf[6]
            self.lower = recf[5]

    def expanded_dim(self, num_vars):
        """Return the size of a vector of dimension 'dim' after
        an expansion of degree 'self._degree'."""
        return (self.degree+1)**num_vars

    def execute(self, x, check=False):
        """Expansion of the data."""
        if check:
            self.check_domain(x)
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

    def check_domain(self, x, prec=1e-6):
        """Checks for compliance of the data x with the domain on which
            the function sequence selected is defined or orthogonal.
        :param x: The data to be expanded. Observations/samples must
            be along the first axis, variables along the second.
        :type x: numpy.ndarray
        :param prec: (Numerical) tolerance when checking validity.
        :type prec: float
        :raise mdp.NodeException: If one or more values lie outside of the function
            specific domain.
        """
        xmax = np.amax(x)-prec
        xmin = np.amin(x)+prec

        if (self.upper < xmax) or (self.lower > xmin):
            raise Exception(
                "One or more values lie outside of the function specific domain.")


class BasisTensorIndicator(object):
    """Helps manage the indices of an up to 52 dimensional multi-dim array."""

    letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

    def __init__(self, num_vars, reach, first=1):
        # create index array of the last elements
        if num_vars + 1 > len(self.letters):
            raise Exception("Too many variable in order to use the einstein\
                            notation(requires letters).")

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
