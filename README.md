<img align="left" src="https://api.travis-ci.org/NiMlr/PyFunctionBases.svg?branch=master">

<img align="right" width="300" height="300" src="https://user-images.githubusercontent.com/39880630/56446422-dc61eb80-6302-11e9-8b46-78c0a9d08420.gif">

# PyFunctionBases
A Python module to compute multi-dimensional arrays of evaluated functions based on Numpy. This module can be used for evaluation of functions, approximation or for feature engineering in machine learning.

Specifically, the module evaluates basis functions on intervals by employing a recursive formula of type
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?f_{n&plus;1}(x)&space;=&space;g(f_n(x),&space;\dots,&space;f_0(x),x)." title="f_{n+1}(x) = g(f_n(x), \dots, f_0(x),x)." />
</p>

This is generalized to the multi-dimensional case by using a tensor product
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?(f_i({x_m}_k),f_j({x_m}_l))&space;\mapsto&space;f_i({x_m}_k)f_j({x_m}_l)" />
</p>

repeatedly on coordinate wise one-dimensional function bases. The code is vectorized over the evalution points
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?x_m&space;\in&space;\mathbb{R}^{num\_dim},&space;m&space;\in&space;\{1,&space;\dots,&space;num\_samples\}" />
</p>

and returns a multi-dimensional array of shape `(num_samples, degree+1, ..., degree+1)`, where `degree`
is the cardinality of the one-dimensional bases omitting a constant function. The following picture shows the two-dimensional case.

<p align="center">
<img width="399" height="323" src="https://user-images.githubusercontent.com/39880630/56447919-80e82b80-630b-11e9-92bd-6d81b0d78946.png">
</p>

Currently, the following functions are available:


| Name | Domain |  
|-------|-----------|
| [`standard_poly`](https://en.wikipedia.org/wiki/Polynomial) | `(-Inf, Inf)`|
| [`legendre_poly`](https://en.wikipedia.org/wiki/Legendre_polynomials) | `[-1, 1]`|
| [`legendre_rational`](https://en.wikipedia.org/wiki/Legendre_rational_functions) | `[0, Inf)`|
| [`chebyshev_poly`](https://en.wikipedia.org/wiki/Chebyshev_polynomials#First_kind) | `[-1, 1]`|

Please make sure that your data lies in these domains, checks will be run if desired.

### Contents
[1. Installation](#installation)  
[2. Simple usage](#simple-usage)  
[3. Where evaluation of polynomials can fail](#where-evaluation-of-polynomials-can-fail)  

## Installation 
Requirements: `pip3 install numpy`

```bash
pip3 install pyfunctionbases
```


## Simple Usage
Now a simple example using standard polynomials is given. By exchanging the name parameter, you can try different functions.

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
f_ij = expn.execute(x)

# flatten the result if needed
f_k = f_ij.reshape(num_samples,(degree+1)**num_dim)
```

## Where evaluation of polynomials can fail
When evaluating functions it is easy to encounter numerical pitfalls. For polynomials specifically one can take measures to avoid problems with floating point numbers, e.g. by employing the representation indicated on the right hand side of the equation `c_1*(x**2)+ c_0*x = x*(c_1*x +c_0)`. Generalizing the former, one can avoid unnecessarily large or small numbers during the evaluation that are caused by large powers and which are badly represented by floating point numbers.

In approximation on the other hand, a basis representation like ``[x**n, ..., x**0]`` is useful in search for the right coefficients. This is a case where e.g. Legendre polynomials provide a useful alternative basis, that covers the exact same function space when the same degrees are considered. In the following code snipped, we can observe an example of this.

![approx](https://user-images.githubusercontent.com/39880630/56443826-8d15be00-62f6-11e9-9cc2-43ae51ed8376.gif)

```python
from pyfunctionbases import RecursiveExpansion
import numpy as np
import matplotlib.pyplot as plt

# create some data
samples = 1000
x = np.random.uniform(low=-1.0, high=1.0, size=(samples,))
x.sort()
# evaluate a function to approximate on the data
fvals = np.tanh(x)*np.cos(50*x)

# set some a maximum degree for the polynomials
degree = 50

# initialize the RecursiveExpansion
expnleg = RecursiveExpansion(degree, recf='legendre_poly')
expnstan = RecursiveExpansion(degree, recf='standard_poly')

# compute the basis functions
basisleg = expnleg.execute(x[:, None], prec=1e-6)
basisstan = expnstan.execute(x[:, None], prec=1e-6)

# find the coefficients of the least squares fit
# to the function given the data
solleg = np.linalg.lstsq(basisleg, fvals, rcond=None)
solstan = np.linalg.lstsq(basisstan, fvals, rcond=None)

# plot the result
plt.plot(x, fvals)
plt.plot(x, np.matmul(basisleg, solleg[0]))
plt.plot(x, np.matmul(basisstan, solstan[0]))
plt.show()
```
