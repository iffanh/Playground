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
import itertools
import scipy
import scipy.special
import functools
from scipy.optimize import minimize, NonlinearConstraint, Bounds
from typing import Any, Tuple, List
import sympy as sp
from sympy import lambdify

from generate_poised_sets import SampleSets

class PolynomialBase:
    def __init__(self, symbol, feval:callable) -> None:
        
        self.symbol = symbol
        self.feval = feval
        
class LagrangePolynomial:
    def __init__(self, symbol, feval:callable) -> None:
        
        self.symbol = symbol
        self.feval = feval
        self.max_sol = None
        
    def _func_to_minimize(self, x:np.ndarray):
        return -np.abs(self.feval(*x))
    
    def cons_f(self, x:np.ndarray, center:np.ndarray) -> float:
        return np.linalg.norm(x - center)
    
    def _define_nonlinear_constraint(self, rad:float, center:np.ndarray) -> NonlinearConstraint:
        return NonlinearConstraint(functools.partial(self.cons_f, center), 0, rad)
    
    def _define_nonlinear_bounds(self, rad:float, center:np.ndarray) -> Bounds:
        return Bounds(lb=center-rad, ub=center+rad)
        
    def _find_max_given_boundary(self, x0:np.ndarray, rad:float, center:np.ndarray) -> Tuple[Any, float]:
        
        if self.max_sol:
            pass
        else:
            nlinear_bounds = self._define_nonlinear_bounds(rad, center)
            nlinear_constraint = self._define_nonlinear_constraint(rad, center)
            self.max_sol = minimize(self._func_to_minimize, x0, method='SLSQP', bounds=nlinear_bounds, constraints=[nlinear_constraint])
        
        self.min_lambda = self.feval(*self.max_sol.x)
        
        return self.max_sol.x, self.min_lambda
        
class ModelPolynomial:
    def __init__(self, symbol, feval:callable) -> None:
        
        self.symbol = symbol
        self.feval = feval
        
class Poisedness:
    def __init__(self, index, max_points, poisedness) -> None:
        self.index = index # Index with the maximum poisedness
        self.max_points = max_points
        self.poisedness = poisedness
        
        self._validate_class()
        
    def _validate_class(self):
        assert np.max(self.poisedness) == self.poisedness[self.index]
    
    def max_poisedness(self):
        return np.max(self.poisedness)
    
    def point_to_max_poisedness(self):
        return self.max_points[self.index]
    

class LagrangePolynomials:
    def __init__(self, pdegree:int = 2):
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
            .get_poisedness : Calculate (minimum) poisedness of the given set
            
        """
        ### Must exists
        self.pdegree = pdegree
        
        
    def initialize(self, v:np.ndarray, f:np.ndarray, sort_type:str='function', lpolynomials:List[LagrangePolynomial] = None):
        self.sample_set = SampleSets(v, sort_type=sort_type, f=f)
        
        self.y = self.sample_set.y
        self.f = f[self.sample_set.sorted_index]
        
        self.N = v.shape[0]
        self.P = v.shape[1]
        
        self.multiindices = self._get_multiindices(self.N, self.pdegree, self.P)
        self.coefficients = self._get_coefficients(self.multiindices)
        
        # Possible improvement is to let user decide which basis they want, e.g. for underdetermined problems (Ch. 5)
        self.polynomial_basis, self.input_symbols = self._get_polynomial_basis(self.multiindices, self.coefficients)
        
        if lpolynomials is None:
        self.lagrange_polynomials = self._build_lagrange_polynomials(self.polynomial_basis, self.y, self.input_symbols)
        else: # When lagrange polynomials is constructed manually
            self.lagrange_polynomials = lpolynomials
             
        self.model_polynomial = self._build_model_polynomial(self.lagrange_polynomials, self.f, self.input_symbols)
        
        self.gradient, self.Hessian = self._get_coefficients_from_expression(self.model_polynomial.symbol, self.input_symbols, self.pdegree)    
        
        self.index_of_largest_lagrangian_norm = None
        
    def interpolate(self, x:np.ndarray) -> float:
        """Interpolate using the interpolation model given the input x. 
           Maps R^n -> R, where n is the dimension of input data

        Args:
            x (np.ndarray): Input data, R^n

        Returns:
            float: Interpolation value
        """
        return self.model_polynomial.feval(*x)
        
    def poisedness(self, rad=None, center=None) -> Poisedness:
        """Calculate the minimum poisedness given the set of interpolation points.'
           The poisedness is calculated as: 
           
                Lambda = max_{0 <= i <= p} max_{x \in B} |lambda_i(x)|

        Returns:
            int: index of lagrange polynomial with the largest poisedness
            float: Minimum poisedness of the given interpolation set
        """
        
        ## rad and center should only be used if poisedness is done "deliverately" in Ball(center, rad)
        ## in most cases rad and center should be default to None and therefore the rad and center
        ## will be calculated directly from the lagrange polynomial's sample set
        ## only in special cases that rad and center are specified, like in the case of Algorithm 6.3
        if rad is None:
            rad = self.sample_set.ball.rad
        if center is None:
            center = self.sample_set.ball.center
        
        return self._get_poisedness(self.lagrange_polynomials, rad, center)
        
    def _get_poisedness(self, lagrange_polynomials:List[LagrangePolynomial], rad:float, center:np.ndarray) -> Poisedness:
        
        Lambda = 0.0
        index = 0
        
        Lambdas = []
        max_sols = []
        
        for i, lp in enumerate(lagrange_polynomials):
            max_sol, feval = lp._find_max_given_boundary(x0=self.y[:,0], rad=rad, center=center)          
            max_sols.append(max_sol)
            Lambdas.append(np.abs(feval))
            
            if np.abs(feval) > Lambda:
                Lambda = np.abs(feval)
                index = i
            
        if Lambda == 0:
            raise Exception(f"Poisedness (Lambda) is 0. Something is wrong.")
        
        return Poisedness(index, max_sols, Lambdas)
    
    def _construct_b_apprx(self, b1:float, b2:float) -> np.ndarray:
        return np.array([b1, b2])
    
    def _construct_c_apprx(self, c11:float, c12:float, c21:float, c22:float) -> np.ndarray:
        return np.array([[2*c11, 2*c12], [2*c21, 2*c22]])
    
    def _construct_gradient(self, b1:float, b2:float, c11:float, c12:float, c21:float, c22:float) -> callable:
        return lambda x: self._construct_b_apprx(b1, b2) + np.matmul(self._construct_c_apprx(c11, c12, c21, c22), x)
    
    def _construct_Hessian(self, c11:float, c12:float, c21:float, c22:float) -> np.ndarray:
        return self._construct_c_apprx(c11, c12, c21, c22)
    
    def _get_coefficients_from_expression(self, expression, input_symbols, degree:int=2) -> Tuple[np.ndarray, np.ndarray]:
        """ The function responsible for approximating the gradient and Hessian

        Args:
            expression (_type_): _description_
            input_symbols (_type_): _description_
            degree (int, optional): _description_. Defaults to 2.

        Raises:
            Exception: _description_

        Returns:
            Tuple[np.ndarray, np.ndarray]: _description_
        """
        if degree == 2:
            x1, x2 = input_symbols
            x1p2, x2p2, x1x2 = x1**2, x2**2, x1*x2
            
            b1 = lambdify(x2, expression.coeff(x1), 'numpy')(0); b1 = self._is_close_to_zero(b1)
            b2 = lambdify(x1, expression.coeff(x2), 'numpy')(0); b2 = self._is_close_to_zero(b2)
            c11 = expression.coeff(x1p2); c11 = self._is_close_to_zero(c11)
            c22 = expression.coeff(x2p2); c22 = self._is_close_to_zero(c22)
            c12 = expression.coeff(x1x2)/2; c12 = self._is_close_to_zero(c12) # We only know c12 + c21 = constant
            c21 = expression.coeff(x1x2)/2; c21 = self._is_close_to_zero(c21)

            gradient = self._construct_gradient(b1, b2, c11, c12, c21, c22) 
            Hessian = self._construct_Hessian(c11, c12, c21, c22)
            
        elif degree == 1:
            x1, x2 = input_symbols
            
            b1 = expression.coeff(x1); b1 = self._is_close_to_zero(b1)
            b2 = expression.coeff(x2); b2 = self._is_close_to_zero(b2)
            
            gradient = self._construct_b_apprx(b1, b2) 
            Hessian = None
        
        else: 
            raise Exception(f"Polynomial degree of more than 2 is not yet supported.")
        
        return (gradient, Hessian)
    
    def _is_close_to_zero(self, var:float, tol:float=10E-5):
        if np.abs(var) < tol:
            return 0.0
        else:
            return float(var)
    
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
            
        return ModelPolynomial(polynomial_sum, lambdify(list(input_symbols), polynomial_sum, 'numpy'))

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
        basis_matrix = sp.Matrix(matrix)
        basis_vector = sp.Matrix([d.symbol for d in basis])
        lagrange_matrix = basis_matrix*(basis_matrix.T*basis_matrix).pinv()*basis_vector # Eq (4.7)
        
        lpolynomials = []
        for i in range(lagrange_matrix.shape[0]):
            lpolynomial = lagrange_matrix[i, :][0]
            lpolynomials.append(LagrangePolynomial(lpolynomial, lambdify(input_symbols, lpolynomial, 'numpy')))
        
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
        input_symbols = [sp.symbols('x%s'%(ind+1)) for ind in range(Ndim)]
        basis = []
        for (exps, coef) in zip(exponents, coefficients):
            b = 1
            for symb, exp in zip(input_symbols, exps):
                b *= symb**exp
            b = b/coef
            
            # (phi(x) sympy symbol, phi(x) evaluation), function evaluation called using a numpy array x, phi(*x)
            basis.append(PolynomialBase(b, lambdify(input_symbols, b, 'numpy'))) 

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
            list: all possible combination of the multiindices, Combination(n + d, d)
        """
        range_tuples = tuple([range(d + 1)]*n)
        multiindices = self._product(*range_tuples)
         
        multiindices = [ind for ind in multiindices if np.sum(ind) <= d]
        
        if p < len(multiindices):
            raise Exception(f"The length of interpolation points: {p} is less than the minimum requirement : {len(multiindices)}")
        
        return multiindices 
         
    def _product(self, *argument):
        return itertools.product(*argument)
    
    def _check_lagrange_polynomials(self):
        """ Sanity check on the lagrange polynomial condition. Testing the kroenecker delta condition.
        """
        for i, polynomial in enumerate(self.lagrange_polynomials):
            for j in range(self.y.shape[1]):
                eval = polynomial.feval(*self.y[:, j])
                
                if i == j:
                    assert np.abs(eval - 1) <= 10E-5
                else:
                    assert np.abs(eval) <= 10E-5
    
class ModelImprovement:
    """ Class that responsible for improving the lagrange polynomial models based on the poisedness of set Y. 
    
    """
    def __init__(self) -> None:
        pass
    
    def improve_model(self, lpolynomials:LagrangePolynomials, func:callable, L:float=100.0, max_iter:int=5, sort_type='function') -> LagrangePolynomials:
        """ The function responsible for improving the poisedness of set Y in lagrange polynomial. 
        It follows from Algorithm 6.3 in Conn's book.

        Args:
            lpolynomials (LagrangePolynomials): LagrangePolynomials object to be improved
            func (callable): function call to evaluate the new points
            L (float, optional): Maximum poisedness in the new LagrangePolynomial object. Defaults to 100.0.
            max_iter (int, optional): Number of loops to get to the improved poisedness. Defaults to 5.

        Returns:
            LagrangePolynomials: New LagrangePolynomial object with improved poisedness
        """
        rad = lpolynomials.sample_set.ball.rad
        center = lpolynomials.sample_set.ball.center
        
        for k in range(max_iter):
            # Algorithm 6.3
            poisedness = lpolynomials.poisedness(rad=rad, center=center)
            Lambda = poisedness.max_poisedness()
            if k == 0:
                best_polynomial = lpolynomials
                curr_Lambda = Lambda

            # main loop
            if Lambda > L:
                new_point = poisedness.point_to_max_poisedness()
                new_y = lpolynomials.y*1
                new_y[:, poisedness.index] = new_point
                feval = func(new_point)
                new_f = lpolynomials.f*1
                new_f[poisedness.index] = feval
                lpolynomials = LagrangePolynomials(v=new_y, f=new_f, pdegree=2, sort_type=sort_type)
            else:
                lpolynomials = LagrangePolynomials(v=lpolynomials.y, f=lpolynomials.f, pdegree=2, sort_type=sort_type)
                break
            
            # save polynomial with the smallest poisedness
            if Lambda < curr_Lambda:
                curr_Lambda = Lambda
                best_polynomial = lpolynomials
                
            if k == max_iter-1:
                print(f"Could not construct polynomials with poisedness < {L} after {max_iter} iterations. Consider increasing the max_iter.")
            
        lpolynomials._check_lagrange_polynomials()    
        
        return best_polynomial
    
    def _check_lagrange_polynomials(self, y:np.ndarray, polynomials:List[LagrangePolynomial]):
        
        for i, polynomial in enumerate(polynomials):
            for j in range(y.shape[1]):
                eval = polynomial.feval(*y[:, j])
                
                if i == j:
                    assert np.abs(eval - 1) <= 10E-5
                else:
                    assert np.abs(eval) <= 10E-5