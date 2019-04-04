# PyFunctionBases
A Python module to compute multidimensional arrays of evaluated functions based on Numpy.
Inline `code` has `back-ticks around` it.
Specifically, the module evaluates basis functions on intervals by employing a recursive formula of type
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?f_{n&plus;1}(x)&space;=&space;g(f_n(x),&space;\dots,&space;f_0(x),x)." title="f_{n+1}(x) = g(f_n(x), \dots, f_0(x),x)." />
</p>

This is generalized to the multi-dimensional case by using a tensor product
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?(x,y)&space;\mapsto&space;f(x)g(y)" />
</p>
repeatedly on coordinate wise one-dimensional function bases. The code vectorized over the

`num_samples`

evalution points

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?x_i&space;\in&space;\mathbb{R}^{num\_dim},&space;i\in&space;\{1,&space;\dots,&space;num\_samples\}" />
</p>
and returns a multi-dimensional array of shape 
`(num_samples, degree+1, ..., degree+1)`
, where 
`degree`
is the cardinality of the one-dimensional bases omitting a constant function. Currently, the following functions are available:


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
