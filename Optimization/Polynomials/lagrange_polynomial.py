""" 
    This script should be able to find the polynomials, 
    given the interpolation set Y = {y0, y1, ..., yp} in R^n
    
    The polynomials takes the form:
     - Basis : 
        m(x) = \sum_{i=0}^p \phi_i(x) \alpha_i
     - Lagrange :
        m(x) = \sum_{i=0}^p f(y^i) l_i(x)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import scipy.special
from typing import Tuple, List
import sympy as sp
from sympy import lambdify

class PolynomialBase:
    def __init__(self, symbol, feval:callable) -> None:
        
        self.symbol = symbol
        self.feval = feval
        
class LagrangePolynomial:
    def __init__(self, symbol, feval:callable) -> None:
        
        self.symbol = symbol
        self.feval = feval
        
class ModelPolynomial:
    def __init__(self, symbol, feval:callable) -> None:
        
        self.symbol = symbol
        self.feval = feval


class LagrangePolynomials:
    def __init__(self, v:np.ndarray, f:np.ndarray, pdegree:int = 2):
        """ This class should be able to generate lagrange polynomials given the samples

        Args:
            v (np.ndarray): an N x (P+1) array in which each column j in P represent an interpolation point. 
                            N is the dimension of the vectors.
            f (np.ndarray): an N x 1 array such that f = M \alpha
            pdegree (int) : polynomial degree for the approximation
            
        Methods:
            .interpolate : Given a vector x it will interpolate using the polynomials constructed 
                           by this class
                           
        Attributes: 
            .lagrange_polynomials : All the lagrange polynomials, lambda_i(x), wrapped in LagrangePolynomial class
            .model_polynomial : Ultimate polynomial model for the given interpolation set, wrpaped in ModelPolynomial class
            .polynomial_basis : List of the polynomial basis, wrapped in PolynomialBase class
        
        """

        self.f = f
        self.N = v.shape[0]
        self.P = v.shape[1]
        
        self.multiindices = self._get_multiindices(self.N, pdegree, self.P)
        self.coefficients = self._get_coefficients(self.multiindices)
        self.polynomial_basis, self.input_symbols = self._get_polynomial_basis(self.multiindices, self.coefficients)
        self.lagrange_polynomials = self._build_lagrange_polynomials(self.polynomial_basis, v, self.input_symbols)
        self.model_polynomial = self._build_model_polynomial(self.lagrange_polynomials, f, self.input_symbols)
        
    def interpolate(self, x:np.ndarray) -> float:
        """Interpolate using the interpolation model given the input x. 
           Maps R^n -> R, where n is the dimension of input data

        Args:
            x (np.ndarray): Input data, R^n

        Returns:
            float: Interpolation value
        """
        return self.model_polynomial.feval(*x)
        
    def _build_model_polynomial(self, lagrange_polynomials:List[LagrangePolynomial], f:np.ndarray, input_symbols:list) -> ModelPolynomial:
        """ Generate model polynomial, m(x), given the lagrange polynomials and the interpolation set value

            m(x) = sum(lambda_i(x)*f(y_i))
            
        Args:
            lagrange_polynomials (List[LagrangePolynomial]): List of lagrange polynomials
            f (np.ndarray): interpolation set value
            input_symbols (list): _description_

        Returns:
            ModelPolynomial: Model polynomial
        """
        
        polynomial_sum = 0 
        for i in range(len(lagrange_polynomials)):
            polynomial_sum += lagrange_polynomials[i].symbol*f[i]
            
        return ModelPolynomial(polynomial_sum, lambdify(input_symbols, polynomial_sum, 'numpy'))

    def _build_lagrange_polynomials(self, basis:List[PolynomialBase], data_points:np.ndarray, input_symbols:list) -> List[LagrangePolynomial]:
        """ Responsible for generating the lagrange polynomials using Cramer's rule: 
            lambda(x) = det(M(Y_i, \phi))/det(M(Y, \phi))
            
            where Y_i = Y \ y_i \cup x

        Args:
            basis (List[PolynomialBase]): List of polynomial base
            data_points (np.ndarray): matrix of interpolation set input Y {y1, y2, ..., yp}
            input_symbols (list): list of sympy symbol for the input X = {x1, x2, ..., xn}

        Returns:
            List[LagrangePolynomial]: Lagrange polynomials, {lambda_0, lambda_1, ..., lambda_p}
        """
        
        # Create the full matrix
        matrix = []
        for i in range(data_points.shape[1]):
            input = data_points[:,i]
            entry = [d.feval(*input) for d in basis]
            matrix.append(entry)
        self.full_matrix = sp.Matrix(matrix)
        full_det = self.full_matrix.det()
        
        # create the var matrix: matrix in which for each set i in the interpolation set, 
        # create another set of Y_i = Y \ y_i \cup x
        self.var_matrices = []
        lpolynomials = []
        for i in range(data_points.shape[1]):
            var_matrix = matrix[:]
            var_matrix[i] = [d.feval(*input_symbols) for d in basis]
            var_matrix = sp.Matrix(var_matrix)
            self.var_matrices.append(var_matrix)
            
            lpolynomial = var_matrix.det()/full_det
            lpolynomials.append(LagrangePolynomial(lpolynomial, lambdify(input_symbols, lpolynomial, 'numpy')))
        
        print(f"Lagrange polynomials are: {[p.symbol for p in lpolynomials]}")
        
        return lpolynomials
    
    def _get_polynomial_basis(self, exponents:list, coefficients:list) -> Tuple[List[PolynomialBase], list]:
        """Responsible for generating all the polynomial basis, also returning input_symbols

        Args:
            exponents (list): combination of all possible exponents
            coefficients (list): combination of all possible coefficients

        Returns:
            Tuple[List[PolynomialBase], list]: List of polynomial bases and input symbols
        """

        Ndim = len(exponents[0])
        print(f"Dimension length, N = {Ndim}")
        input_symbols = [sp.symbols('x%s'%(ind+1)) for ind in range(Ndim)]
        print(f"Symbols = {input_symbols}")
        basis = []
        for (exps, coef) in zip(exponents, coefficients):
            b = 1
            for symb, exp in zip(input_symbols, exps):
                b *= symb**exp
            b = b/coef
            
            # (phi(x) sympy symbol, phi(x) evaluation), function evaluation called using a numpy array x, phi(*x)
            basis.append(PolynomialBase(b, lambdify(input_symbols, b, 'numpy'))) 

        print(f"The polynomial basis (phi) are: {[d.symbol for d in basis]}")

        return basis, input_symbols

    def _get_coefficients(self, multiindices:list) -> list:
        """Generate coefficients for every possible polynomial basis

        Args:
            multiindices (list): list of indices that will be used as the reciprocal in the coefficient calculation

        Returns:
            list: all possible combination of the coefficients
        """
        coefficients = [np.multiply(*scipy.special.factorial(ind)) for ind in multiindices]
        return coefficients
    
    def _get_multiindices(self, n:int, d:int, p:int) -> list:
        """ This function is responsible to get the multiindices to build the polynomial basis

        Args:
            n (int): dimension of the data 
            d (int): degree of polynomials
            p (int): number of data points

        Returns:
            list: all possible combination of the multiindices
        """
        range_tuples = tuple([range(d + 1)]*n)
        multiindices = self._product(*range_tuples)
         
        multiindices = [ind for ind in multiindices if np.sum(ind) <= d]
        
        if self.P < len(multiindices):
            raise Exception(f"The length of interpolation points: {p} is less than the minimum requirement : {len(multiindices)}")
        elif self.P > len(multiindices):
            raise Exception(f"The length of interpolation points: {p} is greater than the minimum requirement : {len(multiindices)}")
        
        return multiindices 
         
    def _product(self, *argument):
        return itertools.product(*argument)