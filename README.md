# PyFunctionBases
A Python module to compute multidimensional Arrays of evaluated functions.


[1. Installation](#installation)  
[2. Simple Usage](#simple-usage)  

## Installation:
1. the stable version:  
`pip3 install pyfunctionbases`


## Simple Usage:
```python
from pyfunctionbases import RecursiveExpansion
import numpy as np

# create some data
samples = 1000
num_dim = 2
x = np.random.uniform(low=0.0, high=1.0, size=(samples, num_dim))

# create an expansion object where name can be any
# function name, that is in the table below
degree = 10
name = 'standard_poly'
expn = RecursiveExpansion(degree, recf=name)

# evaluate the function, result is of shape (samples, degree+1, degree+1)
f_ij = expn.execute(x, check=True)

# making collapse the result if needed
f_k = f_ij.reshape(samples,(degree+1)**num_dim)
```
