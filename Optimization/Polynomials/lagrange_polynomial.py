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
from typing import Tuple
import sympy as sp
from sympy import lambdify

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
        """

        self.f = f
        self.N = v.shape[0]
        self.P = v.shape[1]
        
        self.multiindices = self._get_multiindices(self.N, pdegree, self.P)
        self.coefficients = self._get_coefficients(self.multiindices)
        self.polynomial_basis = self._get_polynomial_basis(self.multiindices, self.coefficients)
        # self.det, self.matrix = self._build_matrix(v, self.multiindices, self.coefficients)
        # self.alpha = self._solve_for_alpha(self.matrix, f)
        # self._check_interpolation_quality(v, f)
        
    def interpolate(self, x:np.ndarray) -> float:
        """Given the polynomial construction when this class is initialized, it will return the interpolated values

        Args:
            x (np.ndarray): input vector

        Raises:
            Exception: if the dimension doesn't match

        Returns:
            float: interpolated value
        """
        
        if x.shape[0] != len(self.multiindices[0]):
            raise Exception(f"Dimension of x : {x.shape} is incompatible with the dimension to the basis input : {len(self.multiindices[0])}")
        
        result = 0
        for j, alpha in enumerate(self.alpha):
            result += alpha*self._get_basis_func_eval(x, self.multiindices[j], self.coefficients[j])
        
        return result
        
    # def _create_
    
    def _get_polynomial_basis(self, exponents:list, coefficients:list): # -> list(Tuple[callable]):

        Ndim = len(exponents[0])
        print(f"Dimension length, N = {Ndim}")
        self.symbols = [sp.symbols('x%s'%(ind+1)) for ind in range(Ndim)]
        print(f"Symbols = {self.symbols}")
        self.basis = []
        for (exps, coef) in zip(exponents, coefficients):
            b = 1
            for symb, exp in zip(self.symbols, exps):
                b *= symb**exp
            b = b/coef
            
            # (phi(x) sympy symbol, phi(x) evaluation), function evaluation called using a numpy array x, phi(*x)
            self.basis.append((b, lambdify(self.symbols, b, 'numpy'))) 

        print(f"The polynomial basis are: {self.basis}")

        return self.basis

    def node_specific_constant(self, ind:int, v:np.ndarray) -> float: 

        return 
    
    def _build_matrix(self, v:np.ndarray, multiindices:list, coefficients:list) -> Tuple[float, np.ndarray]:
        """This function is responsible for building the M matrix in M \alpha = f

        Args:
            v (np.ndarray): array of interpolation data points
            multiindices (list): list of indices to be used as exponents. The length is equal to the dimension of the data 
            coefficients (list): list of coefficients. The length is equal to the dimension of the data

        Raises:
            Exception: When the constructed matrix is singular

        Returns:
            Tuple[float, np.ndarray]: the determinant and the matrix M
        """
        
        
        matrix = np.zeros((v.shape[1], len(coefficients)))
        for i in range(v.shape[1]): # number of data points
            for j in range(len(coefficients)): # number of possible combination of basis
                matrix[i,j] = self._get_basis_func_eval(v[:,i], multiindices[j], coefficients[j])
        
        det = np.linalg.det(matrix)
        if det == 0.0:
            raise Exception("The matrix M is singular")
        
        return det, matrix

    def _get_basis_func_eval(self, x:np.ndarray, index:list, coeff:float) -> float:
        """ Evaluate the basis function \phi_i(x). Maps R^n -> R

        Args:
            x (np.ndarray): input vector 
            index (list): indices of the exponent, obtained from get_multiindices
            coeff (float): "coefficient" of the basis, obtained from get_coefficients

        Returns:
            float : basis evaluation
        """
        return np.multiply(*np.power(x, index))/coeff
    
    def _get_coefficients(self, multiindices:list) -> list:
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
        
        return multiindices 
         
    def _product(self, *argument):
        return itertools.product(*argument)