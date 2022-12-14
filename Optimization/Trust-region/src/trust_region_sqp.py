import numpy as np
import casadi as ca
from typing import List, Tuple

from .utils.TR_exceptions import IncorrectConstantsException, EndOfAlgorithm
from .utils.simulation_manager import SimulationManager
from .utils.model_manager import SetGeometry, ModelManager, CostFunctionModel, EqualityConstraintModels, InequalityConstraintModels, ViolationModel
from .utils.trqp import TRQP
from .utils.filter import FilterSQP
# from .utils.lagrange_polynomial import LagrangePolynomials

class TrustRegionSQPFilter():
    def __init__(self, constants:dict, dataset:np.ndarray, cf:callable, eqcs:List[callable], ineqcs:List[callable]) -> None:
        
        def _check_constants(constants:dict) -> dict:

            if constants["gamma_0"] <= 0.0:
                raise IncorrectConstantsException(f"gamma_0 has to be larger than 0. Got {constants['gamma_0']}")

            if constants["gamma_1"] <= constants["gamma_0"]:
                raise IncorrectConstantsException(f"gamma_1 must be strictly larger than gamma_0. Got gamma_1 = {constants['gamma_1']} and gamma_0 = {constants['gamma_0']}")

            if constants["gamma_1"] >= 1.0:
                raise IncorrectConstantsException(f"gamma_1 must be strictly less than 1. Got {constants['gamma_1']}")

            if constants["gamma_2"] < 1.0:
                raise IncorrectConstantsException(f"gamma_2 must be larger than or equal to 1. Got {constants['gamma_2']}")

            if constants["eta_1"] <= 0.0:
                raise IncorrectConstantsException(f"eta_1 must be strictly larger than 0. Got {constants['eta_1']}")

            if constants["eta_2"] < constants["eta_1"]:
                raise IncorrectConstantsException(f"eta_2 must be larger than or equal to eta_1. Got eta_1 = {constants['eta_1']} and eta_2 = {constants['eta_2']}")

            if constants["eta_2"] >= 1.0:
                raise IncorrectConstantsException(f"eta_2 must be strictly less than 1. Got {constants['eta_2']}")

            if constants["gamma_vartheta"] <= 0 or constants["gamma_vartheta"] >= 1:
                raise IncorrectConstantsException(f"gamma_vartheta must be between 0 and 1. Got {constants['gamma_vartheta']}") 

            if constants["kappa_vartheta"] <= 0 or constants["kappa_vartheta"] >= 1:
                raise IncorrectConstantsException(f"kappa_vartheta must be between 0 and 1. Got {constants['kappa_vartheta']}")

            if constants["kappa_radius"] <= 0 or constants["kappa_radius"] > 1:
                raise IncorrectConstantsException(f"kappa_radius must be between 0 and 1. Got {constants['kappa_radius']}")

            if constants["kappa_mu"] <= 0:
                raise IncorrectConstantsException(f"kappa_mu must be strictly larger than 0. Got {constants['kappa_mu']}")

            if constants["mu"] <= 0 or constants["mu"] >= 1:
                raise IncorrectConstantsException(f"mu must be between 0 and 1. Got {constants['mu']}")

            if constants["kappa_tmd"] <= 0 or constants["kappa_tmd"] > 1:
                raise IncorrectConstantsException(f"kappa_tmd must be between 0 and 1. Got {constants['kappa_tmd']}")

            if constants["init_radius"] <= 0:
                raise IncorrectConstantsException(f"Initial radius must be strictly positive. Got {constants['init_radius']}")

            return constants

        def _check_constraints(eqcs:List[callable], ineqcs:List[callable]) -> Tuple:
            n_eqcs = len(eqcs)
            n_ineqcs = len(ineqcs)
            
            return n_eqcs, n_ineqcs

        self.constants = _check_constants(constants=constants)
        self.n_eqcs, self.n_ineqcs = _check_constraints(eqcs=eqcs, ineqcs=ineqcs)
        self.sm = SimulationManager(cf, eqcs, ineqcs) # Later this will be refactored for reservoir simulation

        self.dataset = dataset
        
        self.input_symbols = ca.SX.sym('x', self.dataset.shape[0])

        pass

    def __str__(self) -> str:
        return f"TrustRegionSQPFilter(n_eqcs={self.n_eqcs}, n_ineqcs={self.n_ineqcs})"

    def run_simulations(self, Y):

        # run cost function and build the corresponding model
        fY_cf = self.sm.cf.func(Y)
        # do the same with equality constraints
        fYs_eq = []
        for eqc in self.sm.eqcs.eqcs:
            fY = eqc.func(Y)
            fYs_eq.append(fY)

        # do the same with inequality constraints
        fYs_ineq = []
        for ineqc in self.sm.ineqcs.ineqcs:
            fY = ineqc.func(Y)
            fYs_ineq.append(fY)
        
        # # create violation function Eq 15.5.3            
        self.violations = []
        self.violations_eq = []
        self.violations_ineq = []
        for j in range(Y.shape[1]):
            
            v = 0.0
            v_eq = 0.0
            for i in range(self.sm.eqcs.n_eqcs):
                v = ca.fmax(v, ca.fabs(fYs_eq[i][j]))
                v_eq = ca.fmax(v_eq, ca.fabs(fYs_eq[i][j]))
                
            v_ineq = 0.0
            for i in range(self.sm.ineqcs.n_ineqcs):
                # v = ca.fmax(v, -fYs_ineq[i][j])
                v = ca.fmax(v, ca.fmax(0, -fYs_ineq[i][j]))
                v_ineq = ca.fmax(v_ineq, ca.fmax(0, -fYs_ineq[i][j]))
            
            self.violations.append(v)
            self.violations_eq.append(-v_eq)
            self.violations_ineq.append(-v_ineq)
        
        self.violations = np.array(self.violations)
         
        ## Here we reorder such that the center is the best point
        indices = list(range(self.violations.shape[0]))
        nearzero_viol = np.isclose(self.violations, 
                                   np.array([0.0 for i in range(self.violations.shape[0])]), 
                                   atol=1E-5)

        fY_cf_list = list(-fY_cf)
        
        # triples = list(zip(nearzero_viol, self.violations_ineq, fY_cf_list, indices))
        triples = list(zip(self.violations_eq, self.violations_ineq, fY_cf_list, indices))
        triples.sort(key=lambda x:(x[0], x[1], x[2]), reverse=True)
        
        sorted_index = [ind[3] for ind in triples]
        Y = Y[:, sorted_index]
        fY = fY[sorted_index]
        
        fYs_eq = [f[sorted_index] for f in fYs_eq]
        fYs_ineq = [f[sorted_index] for f in fYs_ineq]
                
        m_cf = CostFunctionModel(input_symbols=self.input_symbols, 
                                 Y=Y, 
                                 fY=fY_cf)

        m_eqcs = EqualityConstraintModels(input_symbols=self.input_symbols, 
                                    Y=Y, 
                                    fYs=fYs_eq)

        m_ineqcs = InequalityConstraintModels(input_symbols=self.input_symbols, 
                                              Y=Y, 
                                              fYs=fYs_ineq)
        
        m_viol = ViolationModel(input_symbols=self.input_symbols, m_eqcs=m_eqcs, m_ineqcs=m_ineqcs, Y=Y)

        return ModelManager(input_symbols=self.input_symbols, m_cf=m_cf, m_eqcs=m_eqcs, m_ineqcs=m_ineqcs, m_viol=m_viol)

    def run_single_simulation(self, y:np.ndarray) -> Tuple[float, float]:
        fy = self.sm.cf.func(y)
        
        v_eq = 0
        for eqc in self.sm.eqcs.eqcs:
            fY = eqc.func(y)
            if ca.fabs(fY) > v_eq:
                v_eq = ca.fabs(fY)
            
        v_ineq = 0
        for eqc in self.sm.ineqcs.ineqcs:
            fY = eqc.func(y)
            if -ca.fabs(fY) > v_ineq:
                v_ineq = -ca.fabs(fY)
                
        v = ca.fmax(0, ca.fmax(v_eq, v_ineq))
            
        return fy, v

    def solve_TRQP(self, models:ModelManager, radius:float) -> Tuple[np.ndarray, float, bool]:
        trqp_mod = TRQP(models, radius)
        sol = trqp_mod.sol.full()[:,0]
        is_trqp_compatible = trqp_mod.is_compatible
        radius = trqp_mod.radius
        return sol, radius, is_trqp_compatible
    
    def change_point(self, models:ModelManager, Y:np.ndarray, y_next:np.ndarray, radius:float, replace_type:str) -> np.ndarray:
        
        if replace_type == 'improve_model':
            # Change point with largest poisedness
            poisedness = models.m_cf.model.poisedness(rad=radius, center=Y[:,0])
            # index to replace -> poisedness.index
            new_Y = Y*1
            new_Y[:, poisedness.index] = y_next
        elif replace_type == 'worst_point': 
            worst_index = models.m_cf.model.f.argsort()[-1]
            new_Y = models.m_cf.model.y*1
            new_Y[:, worst_index] = y_next
        
        return new_Y

    def run(self, max_iter=15):
        
        # initialize filter
        self.filter_SQP = FilterSQP(constants=self.constants)
        radius = self.constants['init_radius']
        Y = self.dataset*1

        self.iterates = []
        for k in range(max_iter):

            if radius < self.constants["stopping_radius"]:
                print(f"Found a solution = {Y[:,0]}")
                break
            
            print(f"================={k}=================")
            self.models = self.run_simulations(Y=Y)
            Y = self.models.m_cf.model.y*1
            y_curr = Y[:,0]
            
            iterates = dict()
            iterates['Y'] = Y
            iterates['fY'] = self.models.m_cf.model.f
            iterates['v'] = self.models.m_viol.violations
            iterates['y_curr'] = Y[:,0]
            iterates['radius'] = radius
            iterates['models'] = self.models
            
            self.iterates.append(iterates)
            
            poisedness = self.models.m_cf.model.poisedness(rad=radius, center=Y[:,0])
            print(f"poisedness = {poisedness.max_poisedness()}")
            if poisedness.max_poisedness() > self.constants['L_threshold']:
                print(f"We have to fix the model")
                sg = SetGeometry(input_symbols=self.input_symbols, Y=Y, rad=radius, L=self.constants['L_threshold'])
                sg.improve_geometry()        
                improved_model = sg.model
                self.models = self.run_simulations(Y=improved_model.y)
                Y = improved_model.y
            
            poisedness = self.models.m_cf.model.poisedness(rad=radius, center=Y[:,0])
            print(f"poisedness after improvement = {poisedness.max_poisedness()}")
            
            if k == 0:
                _fy = self.models.m_cf.model.f
                _v = self.models.m_viol.violations
                
                for ii in range(_v.shape[0]):
                    _ = self.filter_SQP.add_to_filter((_fy[ii], _v[ii]))
            
            try:
                y_next, radius, self.is_trqp_compatible = self.solve_TRQP(models=self.models, radius=radius)
                print(f"y_next = {y_next}")
            except EndOfAlgorithm:
                print(f"Impossible to solve restoration step. Current iterate = {Y[:,0]}")
                break
                
            if self.is_trqp_compatible:
                print(f"TRQP Compatible")
                fy_next, v_next = self.run_single_simulation(y_next)
                is_acceptable_in_the_filter = self.filter_SQP.add_to_filter((fy_next, v_next))

                if is_acceptable_in_the_filter:
                    # print(f"Acceptable in the filter")
                    
                    v_curr = self.models.m_viol.feval(y_curr)
                    
                    mfy_curr = self.models.m_cf.model.model_polynomial.feval(y_curr)
                    mfy_next = self.models.m_cf.model.model_polynomial.feval(y_next)
                    fy_curr = self.models.m_cf.model.f[0]
                    
                    rho = (fy_curr - fy_next)/(mfy_curr - mfy_next)
                    print(f"rho = {rho}")
                        
                    if mfy_curr - mfy_next >= self.constants['kappa_vartheta']*(v_curr**2):
                        print(f"With sufficient decrease")
                        
                        if rho < self.constants['eta_1']:
                            print(f"Not good enough model")
                            radius = self.constants['gamma_1']*radius
                            Y = Y*1
                            # Y = self.change_point(self.models, Y, y_next, radius, 'improve_model')

                        else:
                            print(f"Good enough model")
                            
                            if rho >= self.constants['eta_2']:
                                radius = radius*self.constants['gamma_2']
                                Y = self.change_point(self.models, Y, y_next, radius, 'worst_point')
                            else:
                                radius = radius*self.constants['gamma_1']
                                Y = self.change_point(self.models, Y, y_next, radius, 'improve_model')
                                
                    else:
                        print(f"No sufficient decrease")
                        self.filter_SQP.add_to_filter((fy_next, v_next))
                            
                        if rho >= self.constants['eta_2']:
                            radius = radius*self.constants['gamma_2']
                            Y = self.change_point(self.models, Y, y_next, radius, 'worst_point')
                        else:
                            radius = radius*self.constants['gamma_1']
                            Y = self.change_point(self.models, Y, y_next, radius, 'improve_model')
                    
                    pass
                
                else:
                    print(f"Not acceptable in the filter")
                    radius = self.constants['gamma_0']*radius
                    Y = Y*1
            
            else:
                print(f"TRQP Incompatible")
                fy_curr = self.models.m_cf.model.f[0]
                v_curr = self.models.m_viol.feval(y_curr)
                _ = self.filter_SQP.add_to_filter((fy_curr, v_curr))
                
                Y = self.change_point(self.models, Y, y_next, radius, 'improve_model')
                             
        pass