"""
Implementation of the core functionality of the package.

This module the core part of the project, that is, a general class
``RecursiveExpansion`` that builds a multidimensional array of basis functions
for specified data. This is done on a relatively high level using a class
``NDArrayManager``, that helps accessing specific parts of multi-dimensional
Numpy arrays.
"""
import numpy as np
from .function_definitions import recfs


class RecursiveExpansion(object):
    """Multidimensional arrays of recursively computable evaluated
    functions based on Numpy.

    Specifically, the module evaluates basis functions on intervals
    by employing a recursive formula of type

    .. math:: f_{n+1}(x) = g(f_n(x), \dots, f_0(x),x),

    which must be implemented operating on an ND-array and can but is not
    required to be implemented using ``NDArrayManager``. Preimplemented
    are standard polynomials, Legendre polynomials, Legendre rational functions
    and chebychev polynomials of first kind.

    Parameters
    ----------
    degree : int
        A positive integer containing the maximal recursion depth of the
        formula, that is,  :math:`n \in \{0, \dots ,degree\}`.
    recf : str or tuple
        A string of either 'standard_poly', 'legendre_poly', 'legendre_rational',
        'chebychev_poly'. See ``recfs`` in ``function_definitions.py``.
        In the custom case this must be a tuple of type

        (index where recursion
        starts, reach backwards regarding the arguments to the above specified
        function g - this is n+1 in the extreme case, index of a special element
        to be used in the recursion - so you do not have to increase the reach
        to i.e. get the second basis function, initialization function that
        precedes the recursion, recursion function, lower boundary of the
        interval on which your functions are defined, upper boundary of the
        interval on which your functions are defined)

        For more concrete information, see the  ``function_definitions.py``
        module or the examples in the README.
    """

    def __init__(self, degree, recf='standard_poly', dtype=np.dtype('float64')):
        """Initialize a RecursiveExpansionNode."""

        self.degree = degree
        self.dtype = dtype
        # if in dictionary
        if recf in recfs:
            # where the recursion starts
            self.rec_start = recfs[recf][0]
            # number of elements preceding to consider in the recursive step
            self.reach = recfs[recf][1]
            self.special = recfs[recf][2]
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
            self.special = recf[2]
            self.r_init = recf[3]
            self.recf = recf[4]
            self.upper = recf[6]
            self.lower = recf[5]

    def expanded_dim(self, num_dims):
        """Return the number of elements in a multidimensional
        array containing all basis functions to be returned.

        Parameters
        ----------
        num_dims : int or numpy.ndarray
            Dimension of the domain of the multidimensional basis functions
            to compute.

        Returns
        -------
        int or numpy.ndarray
            The number of elements in a multidimensional
            array containing all basis functions to be returned.
        """
        return (self.degree+1)**num_dims

    def execute(self, x, prec=1e-6):
        """Evaluate specified linearily independent (/ orthogonal)
        basis functions with multidimensional domain on data.

        After first generating bases on the factors of
        :math:`\mathbb{R}^{num\_dims}`
        (i.e. each one dimensional subspace) by employing the recursion
        function supplied, the basis is translated to the product-space by
        evaluating a tensor product pairwise using numpy's einsum function.

        Parameters
        ----------
        x : numpy.ndarray
            A two-dimensional numpy array, such that observations of
            variables are stored in the rows of the array, individual variables
            are stored in the columns. This means that one row contains one
            particular observation/measurement of all variables, and one
            column contains all observations/measurements of one particular
            variable.

            Mathematically this means
            :math:`x \in \mathbb{R}^{num_samples \times num_dims}`.
        check : float or None
            Float that indicates a tolerance around the interval boundary
            on which the functions are defined. Set to None if no checks are
            needed.

        Returns
        -------
        numpy.ndarray
            Array containing the evaluation of all generated functions on the
            data points ``x`` supplied. Output is of shape
            ``(num_samples, degree+1, ..., degree+1)``.
        """
        if prec is not None:
            self.check_domain(x, prec=prec)
        deg = self.degree
        num_dims = x.shape[1]
        num_samples = x.shape[0]
        # preset memory
        basetensor = np.empty((num_samples,)+(deg+1,) *
                              num_dims, dtype=self.dtype)
        # initialize index helper
        ind = NDArrayManager(num_dims, self.reach, special=self.special)

        # set elements not in the recursion
        self.r_init(basetensor, ind, x)

        for cur_dim in range(num_dims):
            # preset index for current variable
            ind.zeroAllBut(cur_dim, self.rec_start)
            # single variable recursion
            while ind.getN() <= deg:
                # recursion step
                basetensor[ind.all+ind.getCurrent()] \
                    = self.recf(basetensor, ind, x)
                # next step
                ind.incrementN()

        # inplace tensorproduct
        np.einsum(ind.einsum_notation, *ind.getElementaries(basetensor),
                  out=basetensor, optimize='optimal', dtype=self.dtype)
        return basetensor

    def check_domain(self, x, prec=1e-6):
        """Checks for compliance of the data x with the domain on which
            the function sequence selected is defined.

        Parameters
        ----------
        x : numpy.ndarray
            The data on which the functions will be evaluated.
            Observations/samples must be along the first axis, variables
            along the second. Check the docstring of
            ``RecursiveExpansion.execute`` for a more detailed description.
        prec : float
            (Numerical) tolerance when checking validity.


        Raises
        ------
        Exception
            If one or more values lie outside of the function
            specific domain.
        """
        xmax = np.amax(x)-prec
        xmin = np.amin(x)+prec

        if (self.upper < xmax) or (self.lower > xmin):
            raise Exception(
                "One or more values lie outside of the function specific domain.")


class NDArrayManager(object):
    """Helps manage the indices of an up to 52 dimensional multi-dim array.

    This class helps manage the indices of an up to 52 dimensional
    multi-dim array making use of several get and set methods for
    the highly frequented parts of the array. The limit of 52 is due to
    the Einstein-notation employing characters and at this time
    close to impossible to exceed - even in the simplest case.
    Having this a set of multiindices ``NDArrayManager.indices`` and 
    the following set of methods available greatly increases the ease
    of handling multi-dimensional arrays and reduces the code
    needed to specify the recursion formulas.

    Attributes
    ----------
    indices : numpy.ndarray
        Array of shape ``(reach+1, num_dims)``. Where each row corresponds
        to a multiindex that is adapted using the methods of this class.
        The first row is reserved for the index of a special function such as
        the first order monomial in standard monomials. The remaining ones
        are reserved for the backwards reaching indexes of the evaluated
        functions that are arguments to the recursion.
    """

    letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

    def __init__(self, num_dims, reach, special=1):
        """Initialize an ``NDArrayManager``.

        Parameters
        ----------
        num_dims : int or numpy.ndarray
            Dimension of the domain of the multi-dimensional basis functions
            to compute.
        reach : int
            Number of function values preceding+current one to consider in the
            recursive step, for Legendre polynomials for example 2+1 works.
        special : int
            Saves an index for the index of a function value in the
            multi-dim array to be an argument to the recursive function. This
            way there is no need to increase the reach if only one "special"
            function in the sequence is needed.
        """
        # create index array of the last elements
        if num_dims + 1 > len(self.letters):
            raise Exception("Too many dimensions in order to use the einstein\
                            notation(requires letters).")

        self.indices = np.zeros((reach+1, num_dims), dtype=int)
        self.num_dims = num_dims
        # how many last elements are to consider
        self.reach = reach
        self.current_dim = 0
        # a special element used in every recursive step (e.g. x)
        self.special = special
        # replaces ":"" when tuple indexing numpy array
        self.all = (slice(None),)

        # generate the einstein summation convention string
        # for numpys einsum function
        self.einsum_notation = ''

        for i in range(num_dims):
            self.einsum_notation += self.letters[0]+self.letters[i+1]+','

        self.einsum_notation = self.einsum_notation[:-1] + \
            '->'+self.letters[:num_dims+1]

    def zeroAllBut(self, current_dim, value=0):
        """Zeros every index besides the column of the current variable
        in the index helper.

        That column is set to contain decreasing indices starting with
        the current iteration step of the recursion representing the
        reach backwards.

        Parameters
        ----------
        current_dim : int
            Index of the current dimension on which the recursion is applied.
        """
        self.indices[:, self.current_dim] = 0
        self.current_dim = current_dim
        self.indices[0, current_dim] = self.special
        for i in range(self.reach):
            self.indices[i+1, self.current_dim] = value-i

    def incrementN(self, value=1):
        """Increments the indices (that is all the indices that reach back)
        by ``value`` in the index helper."""
        self.indices[1:, self.current_dim] += value

    def getCurrent(self):
        """Returns the current multi-index that is a tuple of length
        ``num_dims``.

        Returns
        -------
        tuple
            Returns the current set of indices that is a tuple of length
            ``num_dims`` that represents the multiindex of the current
            recursion step.
        """
        return tuple(self.indices[1, :])

    def getPreceding(self, m):
        """Returns the ``m``-th preceding multi-index that is a tuple of length
        ``num_dims``.

        This method is a generalization of ``getCurrent``.

        Parameters
        ----------
        m : int
            The number of steps the index of interest precedes the current.

        Returns
        -------
        tuple
            Returns the m-th preceding multi-index that is a tuple of length
            ``num_dims``.
        """
        return tuple(self.indices[1+m, :])

    def getSpecial(self):
        """Returns the multi-index that corresponds
        to the special function argument of the recursion. 
        It is a tuple of length ``num_dims``.

        Returns
        -------
        tuple
            Returns the multi-index that corresponds
            to the special function argument of the recursion. 
            It is a tuple of length ``num_dims``.
        """
        return tuple(self.indices[0, :])

    def getN(self):
        """Returns the current recursion step along the current
        dimension.

        Returns
        -------
        int
            Returns the current recursion step along the current
            dimension.
        """
        return self.indices[1, self.current_dim]

    def getElementaries(self, basetensor):
        """Return pointers to the edges of the ``num_dims``-dimensional
        array (the code is vectorized over the strictly speaking remaining
        dimension).

        Before the tensor product these are the only non-zero entries. In the
        algorithm they are needed as input to the ``einsum`` function.

        Parameters
        ----------
        basetensor : numpy.ndarray
            The ``num_dims+1``-dimensional array considered, with samples
            along the first dimension. For each sample we have ``num_dims``
            -dimensional subarrays with non-zero values on the edges.

        Returns
        -------
        tuple
            Return pointers to the edges of the ``num_dims``-dimensional
            array (the code is vectorized over the strictly speaking remaining
            dimension). Return value is a list of ``num_dims`` edges
            each having shape (num_samples, degree+1).
        """
        indexlist = [0]*self.num_dims
        elementaries = []
        for i in range(self.num_dims):
            indexlist[i] = slice(None)
            elementaries.append(basetensor[self.all+tuple(indexlist)])
            indexlist[i] = 0
        return elementaries
