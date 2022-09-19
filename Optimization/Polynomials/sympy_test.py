import sympy as sp
from sympy import lambdify
import numpy as np

exponents = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0)] 
coefficients = [1.0, 1.0, 2.0, 1.0, 1.0, 2.0]

Ndim = len(exponents[0])
print(f"Dimension length, N = {Ndim}")
symbols = [sp.symbols('x%s'%(ind+1)) for ind in range(Ndim)]
print(f"Symbols = {symbols}")

basis = []
for i, (exps, coef) in enumerate(zip(exponents, coefficients)):
    ind = i + 1
    b = 1
    for symb, exp in zip(symbols, exps):
        b *= symb**exp
    b = b/coef

    basis.append(b)

print(f"The polynomial basis are: {basis}")

expr = basis[4]
print(expr, symbols)
f = lambdify(symbols, expr, 'numpy')
print(f(2,3))

vectors = np.array([2,3])
print(f(*vectors))