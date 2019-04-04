# PyFunctionBases
A Python module to compute multidimensional arrays of evaluated functions based on Numpy.

Specifically, the module evaluates basis functions on intervals by employing a recursive formula (i.e. f_{n+1}(x) = f_n(x)*x, for standard monomials). This is generalized to the multi-dimensional case by using the tensor product (x,y) -> f(x)g(y) repeatedly using the bases on coordinate wise one-dimensional function bases. The code vectorized over the num_samples evalution points b_i in R^num_dim and returns a multi-dimensional array of shape (num_samples, degree+1, ..., degree+1), where degree is the cardinality of the one-dimensional bases. Currently, the following functions are available:


[1. Installation](#installation)  
[2. Simple Usage](#simple-usage)  

## Installation:
1. the stable version:  
`pip3 install pyfunctionbases`


## Simple Usage:
```python
from pyfunctionbases import RecursiveExpansion
import numpy as np

# create some data to evaluate basis functions on<
num_samples = 1000
num_dim = 2
x = np.random.uniform(low=0.0, high=1.0, size=(num_samples, num_dim))

# create an expansion object where name can be any
# function name, that is in the table below
degree = 10
name = 'standard_poly'
expn = RecursiveExpansion(degree, recf=name)

# evaluate the function, result is of shape (num_samples, degree+1, degree+1)
f_ij = expn.execute(x, check=True)

# flatten the result if needed
f_k = f_ij.reshape(num_samples,(degree+1)**num_dim)
```
