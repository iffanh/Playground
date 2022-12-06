from .lagrange_polynomial import LagrangePolynomials
from .model_improvement import ModelImprovement
import numpy as np
from typing import Any, Tuple, List
from scipy.optimize import minimize, NonlinearConstraint
import casadi as ca
from casadi import DM, SX
# from model_improvement import ModelImprovement

class EqualityConstraint:
    def __init__(self, polynomial: LagrangePolynomials):
        self.lpolynomial = polynomial
    
    
class SubProblem:
    def __init__(self, polynomial:LagrangePolynomials, radius:float, eq_const) -> None:
        
        # Right now the path is solved from solving the lambda, other options are sufficient decrease using pu and pb
        self.polynomial = polynomial
        self.eq_const = eq_const
        self.sol = self._solve_path(radius, eq_const)
#         self.path = self._solve_path_from_lambda(radius=radius).T

    def _calculate_pb(self, g:np.ndarray, B:np.ndarray) -> np.ndarray:
        return -np.matmul(np.linalg.pinv(B), g).T[0]
    
    def _construct_nominators(self, gradient:np.ndarray, decom_mat:np.ndarray) -> Tuple[np.ndarray, bool]:
        is_hard_case = False
        nominators = []
        for i in range(decom_mat.shape[1]):
            # print(f"decom_mat = {decom_mat[:, i].shape}, gradient = {gradient.shape}")
            nominator = np.matmul(decom_mat[:, i], gradient)
            if np.abs(nominator) < 10E-5:
                is_hard_case = True
            
            nominators.append(nominator)

        return np.array(nominators), is_hard_case
    
    def _construct_function_r(self, eigenval:np.ndarray, nominators:np.ndarray, radius:float) -> callable:
        
        resx = lambda x: np.abs((np.power((eigenval[0]+x)*(eigenval[1]+x), 2)/np.sum([np.power(nominators[0]*(eigenval[1]+x), 2), np.power(nominators[1]*(eigenval[0]+x), 2)])) - (1/np.power(radius, 2)))
        
        return resx
    
    def _construct_lambda_path_function(self, nominators, eigenvals, Q) -> callable:
        return lambda x: np.linalg.norm(((nominators[0]/(eigenvals[0] + x))*Q[:, 0] + (nominators[1]/(eigenvals[1] + x))*Q[:, 1]))
    
    def _solve_path(self, radius:float, eq_const):
        
        lbg = []
        ubg = []
        
        g = []
        
        # ball
        ball_constraint = ca.sqrt(ca.sum1(self.polynomial.input_symbols**2))
        g.append(ball_constraint)
        lbg.append(0.); ubg.append(radius)
        
        # eq_constraints
        for eq in eq_const:
            g.append(eq.model_polynomial.symbol)
            lbg.append(0.); ubg.append(0.)
            
        g = ca.vertcat(*g)
        
        nlp = {'f': self.polynomial.model_polynomial.symbol, 'g':g, 'x':self.polynomial.input_symbols}
        
        opts = dict()
        opts['ipopt.print_level'] = 0
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        
        # inital guess
        w0 = [1 for i in self.polynomial.input_symbols.nz]
        lbw = [-ca.inf for i in self.polynomial.input_symbols.nz]
        ubw = [ca.inf for i in self.polynomial.input_symbols.nz]
        
        res = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

        return res['x'].full()[:, 0]
    
    
class TrustRegion:
    def __init__(self, input_symbols, dataset, results, func:callable, eq_const:List[EqualityConstraint]) -> None:
        
        self.input_symbols = input_symbols
        self.polynomial = LagrangePolynomials(input_symbols=input_symbols, pdegree=2)
        self.polynomial.initialize(v=dataset, f=results)
        self.func = func
        self.sp = None
        
        self.dataset = self.polynomial.y
        self.results = self.polynomial.f
        
        self.rad = self.polynomial.sample_set.ball.rad # radius of trust region
        self.center = self.polynomial.sample_set.ball.center # center of trust region
        
        self.eq_const = eq_const
    
    def run(self, max_radius:float=1.5, max_iter:int=20, rad_tol:float=1E-5):
        
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
        eta1 = 0.25
        gamma = 0.5
        gamma_inc = 1.2
        eps_c = 0.5
        beta = 0.3
        mu = 0.4
        omega = 0.8
        L = 1.1
        status_crit = 'Initializing'
        status_acc = 'Initializing'
        
        
        x0 = self.polynomial.sample_set.ball.center
        gradient = self.polynomial.gradient(x0)[0]
        Hessian = self.polynomial.Hessian
        
        sigma_inc = self.find_sigma(gradient, Hessian)
        m_inc = self.polynomial
        rad_inc = m_inc.sample_set.ball.rad
            
        
        self.list_of_models = []
        self.list_of_radius = []
        self.list_of_status = []
        self.list_of_OF = []
            
        k = 0
        while rad_inc >= rad_tol:
            
            if k > max_iter:
                break
                
            self.list_of_models.append(m_inc)
            self.list_of_radius.append(rad_inc)
            self.list_of_status.append({'criticality_step' : status_crit, 'acceptance': status_acc})
            self.list_of_OF.append(m_inc.f[0])
            # try:
            if True:
                print(f"====================Iteration {k}====================")
                m, rad, _, status_crit = self.criticality_step(m_inc, func, sigma_inc, eps_c, rad_inc, mu, beta, omega, L)
                x_opt = self.step_calculation(m, rad, self.eq_const)
                print(f"point from step calculation = {x_opt}")
                m_inc, rho, sigma, status_acc = self.acceptance_of_the_trial_point(m, func, x0, x_opt, eta1, omega, L, mu, rad)
                rad_inc = self.trust_region_radius_update(rho, rad, gamma_inc, gamma, eta1, beta, sigma, max_radius)
                print(f"Best point = {m_inc.y[:,0]}")
                print(f"Best OF: {m_inc.f[0]}")
                print(f"rho : {rho}")
                print(f"Radius: {rad_inc}")
                print(f"All points : {m.y}")
                print(f"Set poisedness = {m_inc.poisedness(rad=rad_inc, center=m_inc.y[:,0]).poisedness}")
            # except:
            #     print(f"Process terminated due to non-invertible matrix")
            #     break
            
            k += 1
            
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
    
    def acceptance_of_the_trial_point(self, m:LagrangePolynomials, func:callable, x0, x_opt, eta1:float, omega, L, mu, rad:float):
        # Step 3 of Algorithm 10.3
        f1, f2 = func(x0), func(x_opt)
        m1, m2 = m.model_polynomial.feval(x0), m.model_polynomial.feval(x_opt)
        
        rho = self.calculate_rho_ratio(f1, f2, m1, m2)
        
        if rho >= eta1:
        # if True:
            status = "Trial_point: Successful"
            sort_ind = m.f.argsort(axis=0)
            _my = m.y[:, sort_ind]
            _mf = m.f[sort_ind]
            _my[:, -1] = x_opt
            _mf[-1] = func(x_opt)
            
            m_inc = LagrangePolynomials(input_symbols=self.input_symbols, pdegree=2)
            m_inc.initialize(v=_my, f=_mf, sort_type="function", tr_radius=rad)
            
            #  // TODO elif rho >= eta0: we need to first create a function to detect fully linear/quadratic condition
            #     pass
        else:
            # Step 4 in Algorithm 10.3
            status = "Trial_point: Model improving"
            m_inc = LagrangePolynomials(input_symbols=self.input_symbols, pdegree=2)
            m_inc.initialize(v=m.y, f=m.f, sort_type="function", tr_radius=rad)

            center = m_inc.sample_set.ball.center
            
            mi = ModelImprovement(input_symbols=self.input_symbols)
            m_inc, im_status = mi.improve_model(lpolynomials=m_inc, func=func, rad=rad, center=center, L=L, max_iter=10, sort_type='function')
            
            status += f" # points replaced : {im_status['points_replaced']}, reduce radius = {im_status['radius_changed']}"
            
        x0 = m_inc.sample_set.ball.center
        gradient = m_inc.gradient(x0)
        Hessian = m_inc.Hessian
        sigma_inc = self.find_sigma(gradient, Hessian)
            
        return m_inc, rho, sigma_inc, status
    
    def step_calculation(self, m:LagrangePolynomials, rad:float, eq_const) -> np.ndarray:
        # Step 2 of Algorithm 10.3
        
#         optimal_path = self.solve_subproblem(m, rad=rad, eq_const=eq_const).path
#         optimal_solution = m.sample_set.ball.center + optimal_path
        
#         return optimal_solution
        return self.solve_subproblem(m, rad=rad, eq_const=eq_const)
        
    def criticality_step(self, m_inc:LagrangePolynomials, func:callable, sigma_inc:float, eps_c:float, rad_inc:float, mu:float, beta:float, omega:float, L:float) -> Tuple[LagrangePolynomials, float]:
        # Step 1 of Algorithm 10.3
        if sigma_inc <= eps_c and rad_inc > mu*sigma_inc:
            status = 'Criticality : Model improving'
            rad = rad_inc*1
            points_replaced = 0
            radius_changed = False
            while rad > mu*sigma_inc:
                m, rad, sigma, im_status = self._model_improvement(m_inc, func, omega, L, mu, rad_inc)
                rad = np.min([rad_inc, np.max([rad, beta*sigma])])
                
                points_replaced += im_status['points_replaced']
                if im_status['radius_changed']:
                    radius_changed = True
                    
            status += f" # points replaced : {points_replaced}, reduce radius = {radius_changed}"
            
        else:
            status = 'Criticality : Successful'
            m = m_inc
            rad = rad_inc*1
            sigma = sigma_inc*1
        
        return m, rad, sigma, status
    
    def _model_improvement(self, lp:LagrangePolynomials, func:callable, omega:float, L:float, mu:float, rad:float, max_iter:int=10):
        # Algorithm 10.4
        
        rad_k = rad*1
        center = lp.sample_set.ball.center
        for _ in range(max_iter):
            mi = ModelImprovement()
            
            lp, im_status = mi.improve_model(lpolynomials=lp, func=func, rad=rad_k, center=center, L=L, max_iter=10, sort_type='function')
            
            x0 = lp.sample_set.ball.center
            gradient = DM(lp.gradient(x0)[0]).full()
            Hessian = lp.Hessian
            
            sigma_i = self.find_sigma(gradient, Hessian)
            rad_k = omega*lp.sample_set.ball.rad
        
            if rad_k <= mu*sigma_i:
                break
        
        return lp, rad_k, sigma_i, im_status
    
    def find_sigma(self, gradient:np.ndarray, Hessian:np.ndarray):
        eigs = np.linalg.eigh(Hessian)[0]
        try:
            min_eig = np.min(eigs[eigs < 0])
        except ValueError:
            min_eig = 0.
            
        return np.max([ca.norm_2(gradient), -min_eig])
    
    def solve_subproblem(self, polynomial:LagrangePolynomials, rad:float, eq_const) -> SubProblem:
        
        # if rad: 
        self.sp = SubProblem(polynomial, radius=rad, eq_const=eq_const)
        # else:
        #     self.sp = SubProblem(polynomial, radius=self.rad)
        
        return self.sp.sol
    
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
    

    