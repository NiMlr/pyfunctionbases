"""
Function definitions that can be used for the ``RecursiveExpansion`` class.

Within this module you will find the recursive implementations of the
computation of standard polynomials, Legendre polynomials, 
Legendre rational functions and Chebyshev polynomials of first kind.
The advantages of using these functions bases include orthogonality
and numerical stability. Analytically part of the polynomial bases
cover the same function space if they are of same degree and on the
same domain.
"""


def recf_standard_poly(basetensor, ind, x):
    """Implementation of the recursion formula for standard polynomials.

    Parameters
    ----------
    basetensor : numpy.ndarray
        The ``num_dims+1``-dimensional array considered, with samples
        along the first dimension. For each sample we have ``num_dims``
        -dimensional subarrays with non-zero values on the edges.
    ind : numpy.ndarray
        Instance of the class ``NDArrayManager`` created in the
        ``RecursiveExpansion.execute`` method and passed when
        calling this function.
    x : numpy.ndarray
        A two-dimensional numpy array passed to
        ``RecursiveExpansion.execute``. The recursion will be evaluated
        on the data contained. Observations of
        variables are stored in the rows of the array, individual variables
        are stored in the columns. This means that one row contains one
        particular observation/measurement of all variables, and one
        column contains all observations/measurements of one particular
        variable.


    Returns
    -------
    numpy.ndarray
        Array of shape ``(num_samples,)`` containing the evaluation of
        the most recent recursion step along the ``current_dim``.

    References
    ----------
    .. [1] Wikipedia, "Polynomials",
        https://en.wikipedia.org/wiki/Polynomial
    """
    return basetensor[ind.all+ind.getPreceding(1)] * \
        basetensor[ind.all+ind.getSpecial()]


def init_standard_poly(basetensor, ind, x):
    """Initialize the 1 and x1,...,xn before starting the recursion.

    Parameters
    ----------
    basetensor : numpy.ndarray
        The ``num_dims+1``-dimensional array considered, with samples
        along the first dimension. For each sample we have ``num_dims``
        -dimensional subarrays with non-zero values on the edges.
    ind : numpy.ndarray
        Instance of the class ``NDArrayManager`` created in the
        ``RecursiveExpansion.execute`` method and passed when
        calling this function.
    x : numpy.ndarray
        A two-dimensional numpy array passed to
        ``RecursiveExpansion.execute``. The recursion will be evaluated
        on the data contained. Observations of
        variables are stored in the rows of the array, individual variables
        are stored in the columns. This means that one row contains one
        particular observation/measurement of all variables, and one
        column contains all observations/measurements of one particular
        variable.
    """
    ind.zeroAllBut(0, 0)
    basetensor[ind.all+ind.getCurrent()] = 1.
    for i in range(x.shape[1]):
        ind.zeroAllBut(i, 1)
        basetensor[ind.all + ind.getCurrent()] = x[:, i]


def recf_legendre_poly(basetensor, ind, x):
    """Implementation of the recursion formula for Legendre polynomials.

    Parameters
    ----------
    basetensor : numpy.ndarray
        The ``num_dims+1``-dimensional array considered, with samples
        along the first dimension. For each sample we have ``num_dims``
        -dimensional subarrays with non-zero values on the edges.
    ind : numpy.ndarray
        Instance of the class ``NDArrayManager`` created in the
        ``RecursiveExpansion.execute`` method and passed when
        calling this function.
    x : numpy.ndarray
        A two-dimensional numpy array passed to
        ``RecursiveExpansion.execute``. The recursion will be evaluated
        on the data contained. Observations of
        variables are stored in the rows of the array, individual variables
        are stored in the columns. This means that one row contains one
        particular observation/measurement of all variables, and one
        column contains all observations/measurements of one particular
        variable.


    Returns
    -------
    numpy.ndarray
        Array of shape ``(num_samples,)`` containing the evaluation of
        the most recent recursion step along the ``current_dim``.

    References
    ----------
    .. [2] Wikipedia, "Legendre polynomials",
        https://en.wikipedia.org/wiki/Legendre_polynomials
    """
    n = ind.getN()
    return (2.*n-1.)/n*basetensor[ind.all+ind.getSpecial()] \
        * basetensor[ind.all + ind.getPreceding(1)] \
        - (n-1.)/n*basetensor[ind.all + ind.getPreceding(2)]


def init_legendre_rational(basetensor, ind, x):
    """Initialize the initial values before starting the recursion.

    Parameters
    ----------
    basetensor: numpy.ndarray
        The ``num_dims+1``- dimensional array considered, with samples
        along the first dimension. For each sample we have ``num_dims``
        -dimensional subarrays with non-zero values on the edges.
    ind: numpy.ndarray
        Instance of the class ``NDArrayManager`` created in the
        ``RecursiveExpansion.execute`` method and passed when
        calling this function.
    x: numpy.ndarray
        A two-dimensional numpy array passed to
        ``RecursiveExpansion.execute``. The recursion will be evaluated
        on the data contained. Observations of
        variables are stored in the rows of the array, individual variables
        are stored in the columns. This means that one row contains one
        particular observation/measurement of all variables, and one
        column contains all observations/measurements of one particular
        variable.
    """
    ind.zeroAllBut(0, 0)
    basetensor[ind.all+ind.getCurrent()] = 1.
    for i in range(x.shape[1]):
        ind.zeroAllBut(i, 1)
        basetensor[ind.all + ind.getCurrent()] = (x[:, i]-1.)/(x[:, i]+1.)


def recf_legendre_rational(basetensor, ind, x):
    """Implementation of the recursion formula for Legendre rational functions.

    Parameters
    ----------
    basetensor: numpy.ndarray
        The ``num_dims+1``- dimensional array considered, with samples
        along the first dimension. For each sample we have ``num_dims``
        -dimensional subarrays with non-zero values on the edges.
    ind: numpy.ndarray
        Instance of the class ``NDArrayManager`` created in the
        ``RecursiveExpansion.execute`` method and passed when
        calling this function.
    x: numpy.ndarray
        A two-dimensional numpy array passed to
        ``RecursiveExpansion.execute``. The recursion will be evaluated
        on the data contained. Observations of
        variables are stored in the rows of the array, individual variables
        are stored in the columns. This means that one row contains one
        particular observation/measurement of all variables, and one
        column contains all observations/measurements of one particular
        variable.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(num_samples,)`` containing the evaluation of
        the most recent recursion step along the ``current_dim``.

    References
    ----------
    .. [3] Wikipedia, "Legendre rational functions",
        https://en.wikipedia.org/wiki/Legendre_rational_functions
    """
    n = ind.getN()
    xv = x[:, ind.current_dim]
    Rnmin1 = basetensor[ind.all+ind.getPreceding(1)]
    Rnmin2 = basetensor[ind.all+ind.getPreceding(2)]
    return (2.*n-1.)/n*(xv-1.) / (xv+1.) * Rnmin1 - (n-1.) / n * Rnmin2


def recf_chebyshev_poly(basetensor, ind, x):
    """Implementation of the recursion formula for Chebyshev polynomials.

    Parameters
    ----------
    basetensor : numpy.ndarray
        The ``num_dims+1``-dimensional array considered, with samples
        along the first dimension. For each sample we have ``num_dims``
        -dimensional subarrays with non-zero values on the edges.
    ind : numpy.ndarray
        Instance of the class ``NDArrayManager`` created in the
        ``RecursiveExpansion.execute`` method and passed when
        calling this function.
    x : numpy.ndarray
        A two-dimensional numpy array passed to
        ``RecursiveExpansion.execute``. The recursion will be evaluated
        on the data contained. Observations of
        variables are stored in the rows of the array, individual variables
        are stored in the columns. This means that one row contains one
        particular observation/measurement of all variables, and one
        column contains all observations/measurements of one particular
        variable.


    Returns
    -------
    numpy.ndarray
        Array of shape ``(num_samples,)`` containing the evaluation of
        the most recent recursion step along the ``current_dim``.

    References
    ----------
    .. [4] Wikipedia, "Chebyshev polynomials of first kind",
        https://en.wikipedia.org/wiki/Chebyshev_polynomials#First_kind
    """

    return 2. * x[:, ind.current_dim] * basetensor[ind.all+ind.getPreceding(1)] \
        - basetensor[ind.all+ind.getPreceding(2)]


recfs = {'standard_poly': (2, 2, 1, init_standard_poly, recf_standard_poly,
                           -float('Inf'), float('Inf')),
         'legendre_poly': (2, 3, 1, init_standard_poly, recf_legendre_poly,
                           -1, 1),
         'legendre_rational': (2, 3, 0, init_legendre_rational,
                               recf_legendre_rational, 0, float('Inf')),
         'chebyshev_poly': (2, 3, 1, init_standard_poly,
                            recf_chebyshev_poly, -1, 1)}
"""dict: Bundeling the recursions and relevant information.

Each tuple contains the index of the first recursively defined function,
the number of preceding elements that are arguments to the recursion
(+1 for the current), the index of an "special" function that is an argument
to the recursion (may reduce the total amount arguments needed), the
initialization, recursion itself and the boundary of the one-dimensional domain
of the function sequence.
"""
