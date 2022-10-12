from lagrange_polynomial import LagrangePolynomials
import numpy as np
from typing import Any, Tuple, List
from sympy import lambdify
from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint
from model_improvement import ModelImprovement


class SubProblem:
    def __init__(self, polynomial:LagrangePolynomials, radius=2.0) -> None:
        
        # Right now the path is solved from solving the lambda, other options are sufficient decrease using pu and pb
        self.polynomial = polynomial
        self.path = self._solve_path_from_lambda(radius=radius)
        
        
    def _calculate_path(self, tau:float, g:np.ndarray, B:np.ndarray) -> np.ndarray:
        
        if tau <= 1 and tau >= 0:
            path = tau*self._calculate_pu(g, B)
        elif tau <= 2 and tau > 1:
            path = self._calculate_pu + (tau - 1)*self._calculate_pb(g, B)
        else:
            raise Exception(f"tau value has to be between 0 and 2. Got {tau}")
        return path
        
    def _calculate_pu(self, g:np.ndarray, B:np.ndarray) -> np.ndarray:
        a = np.matmul(g.T, g)
        b = np.matmul(np.matmul(g.T, B), g)
        return -(a/b)*g     

    def _calculate_pb(self, g:np.ndarray, B:np.ndarray) -> np.ndarray:
        return -np.matmul(np.linalg.pinv(B), g)
    
    def _construct_nominators(self, gradient:np.ndarray, decom_mat:np.ndarray) -> Tuple[np.ndarray, bool]:
        is_hard_case = False
        nominators = []
        for i in range(decom_mat.shape[1]):
            nominator = np.matmul(decom_mat[:, i], gradient)
            
            if nominator == 0:
                is_hard_case = True
            
            nominators.append(nominator)

        return np.array(nominators), is_hard_case
    
    def _construct_function_r(self, eigenval:np.ndarray, nominators:np.ndarray, radius:float) -> callable:
        
        resx = lambda x: np.abs((np.power((eigenval[0]+x)*(eigenval[1]+x), 2)/np.sum([np.power(nominators[0]*(eigenval[1]+x), 2), np.power(nominators[1]*(eigenval[0]+x), 2)])) - (1/np.power(radius, 2)))
        
        return resx
    
    def _solve_path_from_lambda(self, radius:float):
        
        gradient, Hessian = self.polynomial.gradient(self.polynomial.sample_set.ball.center), self.polynomial.Hessian
        
        if np.linalg.norm(np.matmul(np.linalg.pinv(Hessian), gradient)) <= radius:
            print('Inside the radius')
            path = self._calculate_pb(gradient, Hessian)
        
        else:
            eigenvals, Q = np.linalg.eigh(Hessian) # print(np.matmul(np.matmul(decom_mat, np.diag(eigenvals)), decom_mat.T)) #reconstruction
            nominators, is_hard_case = self._construct_nominators(gradient, Q)
            
            if is_hard_case: 
                sol = -eigenvals[0]
                tau = np.sqrt(radius**2 - (nominators[1]/(eigenvals[1] - eigenvals[0]))**2)
                z = Q[:, 0]
                
                path = - (tau*z + (nominators[1]/(eigenvals[1] + sol.x[0]))*Q[:, 1])
                
            else: # Not hard case
                resx = self._construct_function_r(eigenvals, nominators, radius)
                x0 = -eigenvals[0] + 0.5
                linear_constraint = LinearConstraint(np.ndarray([1]), lb=-eigenvals[0])
                sol = minimize(resx, x0, method='SLSQP', constraints=[linear_constraint])
                
                path = - ((nominators[0]/(eigenvals[0] + sol.x[0]))*Q[:, 0] + (nominators[1]/(eigenvals[1] + sol.x[0]))*Q[:, 1])

        return path
    
class TrustRegion:
    def __init__(self, dataset, results, func:callable) -> None:
        
        self.polynomial = LagrangePolynomials(dataset, results)
        self.func = func
        self.sp = None
        
        self.dataset = self.polynomial.y
        self.results = self.polynomial.f
        
        self.rad = self.polynomial.sample_set.ball.rad # radius of trust region
        self.center = self.polynomial.sample_set.ball.center # center of trust region
        
    
    def run(self, x0:np.ndarray, max_radius:float=2.0):
        
        """ Algorithm: 
        0. Initialization: Set constant variables
            - max_radius = maximum radius allowed in the algorithm
            - init_radius = initial radius of the system
            - eta = threshold of acceptance of new point
        1. Solve subproblem to obtain step p and step length |p|
        2. Calculate new point: x_new = x_center + step
        3. Compute f1 = f(x_center), f2 = f(x_new), m1 = m(x_center), and m2 = m(x_new)
        4. Evaluate rho
        5. Update radius
        6. Update dataset
        """
        
        func = self.func
        
        # Algorithm 10.3
        eta0 = 0.2
        eta1 = 0.5
        gamma = 0.5
        gamma_inc = 1.5
        eps_c = 0.1
        beta = 0.2
        mu = 0.4
        omega = 0.5
        L = 1.2
        
        x0 = self.polynomial.sample_set.ball.center
        gradient = self.polynomial.gradient(x0)
        Hessian = self.polynomial.Hessian
        
        sigma_inc = self.find_sigma(gradient, Hessian)
        m_inc = self.polynomial
        rad_inc = m_inc.sample_set.ball.rad
            
        
        self.list_of_models = []    
            
        for k in range(20):
            
            print(f"========================================")
            m, rad, sigma = self.criticality_step(m_inc, func, sigma_inc, eps_c, rad_inc, mu, beta, omega, L)
            x_opt = self.step_calculation(m)
            m_inc, rho = self.acceptance_of_the_trial_point(m, m_inc, func, x0, x_opt, eta1, omega, L, mu)
            rad_inc = self.trust_region_radius_update(rho, rad, gamma_inc, gamma, eta1, beta, sigma, max_radius)
            
            self.list_of_models.append(m_inc)
            
        return 
    
    def trust_region_radius_update(self, rho, rad, gamma_inc, gamma, eta1, beta, sigma_k, max_radius):
        # Step 5 of Algoriithm 10.3
        
        if rho >= eta1 and rad < beta*sigma_k:
            rad_inc = np.min([gamma_inc*rad, max_radius])
        elif rho >= eta1 and rad >= beta*sigma_k:
            rad_inc = np.min([rad, np.min([gamma_inc*rad, max_radius])])
        elif rho < eta1:
            rad_inc = gamma*rad
        else:
            rad_inc = rad*1
        
        return rad_inc
    
    def acceptance_of_the_trial_point(self, m:LagrangePolynomials, m_inc:LagrangePolynomials, func:callable, x0, x_opt, eta1:float, omega, L, mu):
        # Step 3 of Algorithm 10.3
        f1, f2 = func(x0), func(x_opt)
        m1, m2 = m.model_polynomial.feval(*x0), m_inc.model_polynomial.feval(*x_opt)
        rho = self.calculate_rho_ratio(f1, f2, m1, m2)
        
        if rho >= eta1:
            print("heh")
            sort_ind = m.f.argsort(axis=0)
            m.y = m.y[:, sort_ind]
            m.f = m.f[sort_ind]
            m.y[:, -1] = x_opt
            m.f[-1] = func(x_opt)
            m_inc = LagrangePolynomials(v=m.y, f=m.f, pdegree=2, sort_type="function")
            #  // TODO elif rho >= eta0:
            #     pass
        else:
            print("huh")
            m_inc = LagrangePolynomials(v=m.y, f=m.f, pdegree=2, sort_type="function")
            m_inc, _, _ = self._model_improvement(m_inc, func, omega, L, mu)
            
        return m_inc, rho
    
    def step_calculation(self, m:LagrangePolynomials) -> np.ndarray:
        # Step 2 of Algorithm 10.3
        
        optimal_path = self.solve_subproblem(m).path
        optimal_solution = m.sample_set.ball.center + optimal_path
        
        return optimal_solution
        
    def criticality_step(self, m_inc:LagrangePolynomials, func:callable, sigma_inc:float, eps_c:float, rad_inc:float, mu:float, beta:float, omega:float, L:float) -> Tuple[LagrangePolynomials, float]:
        # Step 1 of Algorithm 10.3
        if sigma_inc <= eps_c:
            if rad_inc > mu*sigma_inc:
                m, rad, sigma = self._model_improvement(m_inc, func, omega, L, mu)

                rad = np.min([rad_inc, np.max([rad, beta*sigma])])
        else:
            m = m_inc
            rad = rad_inc
            sigma = sigma_inc*1
        
        return m, rad, sigma
    
    def _model_improvement(self, lp:LagrangePolynomials, func:callable, omega:float, L:float, mu:float):
        # Algorithm 10.4
        
        for i in range(10):
            mi = ModelImprovement()
            lp2 = mi.improve_model(lpolynomials=lp, func=func, L=L, max_iter=10, sort_type='function')
            
            x0 = lp2.sample_set.ball.center
            gradient = lp2.gradient(x0)
            Hessian = lp2.Hessian
            
            sigma_i = self.find_sigma(gradient, Hessian)
            rad_k = omega*lp2.sample_set.ball.rad
        
            if rad_k <= mu*sigma_i:
                break
        
        return lp2, rad_k, sigma_i
    
    def find_sigma(self, gradient:np.ndarray, Hessian:np.ndarray):
        return np.max([np.linalg.norm(gradient), -np.linalg.eigh(Hessian)[0][-1]])
    
    def solve_subproblem(self, polynomial:LagrangePolynomials, rad=None) -> SubProblem:
        
        if rad: 
            self.sp = SubProblem(polynomial, radius=rad)
        else:
            self.sp = SubProblem(polynomial, radius=self.rad)
        
        return self.sp
    
    def model_evaluation(self, point):
        return self.polynomial.model_polynomial.feval(*point)
    
    def get_new_radius(self, rho:float, step_size:float, max_radius:float, tol = 10E-5) -> float: # Numerical Optimization Algorithm 4.1
        
        if rho < 0.5:
            self.new_rad = 0.5*self.rad
        else: 
            if rho > 0.75 and np.abs(step_size - self.rad) < tol:
                self.new_rad = np.min([2*self.rad, max_radius])
            else:
                self.new_rad = self.rad
        return self.new_rad
    
    def get_new_point(self, rho:float, eta:float) -> np.ndarray:
            
        if rho > eta:
            self.new_point = self.center + self.sp.path
        else:
            self.new_point = self.center
        
        return self.new_point
    
    def calculate_rho_ratio(self, f1:float, f2:float, m1:float, m2:float) -> float:
        
        return (f1 - f2)/(m1 - m2)