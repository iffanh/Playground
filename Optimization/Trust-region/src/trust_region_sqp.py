import numpy as np
import casadi as ca
from typing import List, Tuple

from .utils.TR_exceptions import IncorrectConstantsException
from .utils.simulation_manager import SimulationManager, CostFunction, EqualityConstraints, InequalityConstraints
from .utils.model_manager import ModelManager, CostFunctionModel, EqualityConstraintModels, InequalityConstraintModels, ViolationModel
from .utils.trqp import TRQP
from .utils.filter import FilterSQP

class TrustRegionSQPFilter():
    def __init__(self, constants:dict, dataset:np.ndarray, cf:callable, eqcs:List[callable], ineqcs:List[callable]) -> None:
        
        def _check_constants(constants:dict) -> dict:

            if constants["gamma_0"] <= 0.0:
                IncorrectConstantsException(f"gamma_0 has to be larger than 0. Got {constants['gamma_0']}")

            if constants["gamma_1"] <= constants["gamma_0"]:
                IncorrectConstantsException(f"gamma_1 must be strictly larger than gamma_0. Got gamma_1 = {constants['gamma_1']} and gamma_0 = {constants['gamma_0']}")

            if constants["gamma_1"] >= 1.0:
                IncorrectConstantsException(f"gamma_1 must be strictly less than 1. Got {constants['gamma_1']}")

            if constants["gamma_2"] < 1.0:
                IncorrectConstantsException(f"gamma_2 must be larger than or equal to 1. Got {constants['gamma_2']}")

            if constants["eta_1"] >= 0.0:
                IncorrectConstantsException(f"eta_1 must be strictly larger than 0. Got {constants['eta_1']}")

            if constants["eta_2"] > constants["eta_1"]:
                IncorrectConstantsException(f"eta_2 must be larger than or equal to eta_1. Got eta_1 = {constants['eta_1']} and eta_2 = {constants['eta_2']}")

            if constants["eta_2"] >= 1.0:
                IncorrectConstantsException(f"eta_2 must be strictly less than 1. Got {constants['eta_2']}")

            if constants["gamma_vartheta"] <= 0 or constants["gamma_vartheta"] >= 1:
                IncorrectConstantsException(f"gamma_vartheta must be between 0 and 1. Got {constants['gamma_vartheta']}") 

            if constants["kappa_vartheta"] <= 0 or constants["kappa_vartheta"] >= 1:
                IncorrectConstantsException(f"kappa_vartheta must be between 0 and 1. Got {constants['kappa_vartheta']}")

            if constants["kappa_radius"] <= 0 or constants["kappa_radius"] > 1:
                IncorrectConstantsException(f"kappa_radius must be between 0 and 1. Got {constants['kappa_radius']}")

            if constants["kappa_mu"] <= 0:
                IncorrectConstantsException(f"kappa_mu must be strictly larger than 0. Got {constants['kappa_mu']}")

            if constants["mu"] <= 0 or constants["mu"] >= 1:
                IncorrectConstantsException(f"mu must be between 0 and 1. Got {constants['mu']}")

            if constants["kappa_tmd"] <= 0 or constants["kappa_tmd"] > 1:
                IncorrectConstantsException(f"kappa_tmd must be between 0 and 1. Got {constants['kappa_tmd']}")

            if constants["init_radius"] <= 0:
                IncorrectConstantsException(f"Initial radius must be strictly positive. Got {constants['init_radius']}")

            return constants

        def _check_constraints(eqcs:List[callable], ineqcs:List[callable]) -> Tuple:
            n_eqcs = len(eqcs)
            n_ineqcs = len(ineqcs)
            
            return n_eqcs, n_ineqcs

        self.constants = _check_constants(constants=constants)
        self.n_eqcs, self.n_ineqcs = _check_constraints(eqcs=eqcs, ineqcs=ineqcs)
        self.sm = SimulationManager(cf, eqcs, ineqcs) # Later this will be refactored for reservoir simulation

        self.dataset = dataset

        pass

    def __str__(self) -> str:
        return f"TrustRegionSQPFilter(n_eqcs={self.n_eqcs}, n_ineqcs={self.n_ineqcs})"

    def run_simulations(self, Y):

        input_symbols = ca.SX.sym('x', Y.shape[0])

        # run cost function and build the corresponding model
        fY = self.sm.cf.func(Y)
        m_cf = CostFunctionModel(input_symbols=input_symbols, 
                                 Y=Y, 
                                 fY=fY)

        # do the same with equality constraints
        fYs = []
        for eqc in self.sm.eqcs.eqcs:
            fY = eqc.func(Y)
            fYs.append(fY)
        m_eqcs = EqualityConstraintModels(input_symbols=input_symbols, 
                                          Y=Y, 
                                          fYs=fYs)

        # do the same with inequality constraints
        fYs = []
        for ineqc in self.sm.ineqcs.ineqcs:
            fY = ineqc.func(Y)
            fYs.append(fY)
        m_ineqcs = InequalityConstraintModels(input_symbols=input_symbols, Y=Y, fYs=fYs)
        
        m_viol = ViolationModel(input_symbols=input_symbols, m_eqcs=m_eqcs, m_ineqcs=m_ineqcs, Y=Y)

        return ModelManager(input_symbols=input_symbols, m_cf=m_cf, m_eqcs=m_eqcs, m_ineqcs=m_ineqcs, m_viol=m_viol)

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

    def solve_TRQP(self, models:ModelManager, radius:float):
        trqp_mod = TRQP(models, radius)
        sol = trqp_mod.sol.full()[:,0]
        is_trqp_compatible = trqp_mod.is_compatible
        radius = trqp_mod.radius
        return sol, radius, is_trqp_compatible
    
    def change_point(self, models:ModelManager, Y:np.ndarray, y_next:np.ndarray, radius:float) -> np.ndarray:
        
        # Change point with largest poisedness
        poisedness = models.m_cf.model.poisedness(rad=radius, center=Y[:,0])
        # index to replace -> poisedness.index
        new_Y = Y*1
        new_Y[:, poisedness.index] = y_next
        
        return new_Y

    def run(self, max_iter=5):
        
        # initialize filter
        self.filter_SQP = FilterSQP(constants=self.constants)
        radius = self.constants['init_radius']
        Y = self.dataset*1
        
        
        self.iterates = []
        for k in range(max_iter):
            
            print(f"================={k}=================")
            y_curr = Y[:,0]
            self.iterates.append(Y)
            self.models = self.run_simulations(Y)
            
            poisedness = self.models.m_cf.model.poisedness(rad=radius, center=Y[:,0])
            if poisedness.max_poisedness() > self.constants['L_threshold']:
                print(f"We have to fix the model")
            
            if k == 0:
                _fy = self.models.m_cf.model.f
                _v = self.models.m_viol.violations
                
                for ii in range(_v.shape[0]):
                    _ = self.filter_SQP.add_to_filter((_fy[ii], _v[ii]))
            
            y_next, radius, self.is_trqp_compatible = self.solve_TRQP(models=self.models, radius=radius)
            print(f"y_next = {y_next}")
            
            if self.is_trqp_compatible:
                print(f"TRQP Compatible")
                fy_next, v_next = self.run_single_simulation(y_next)
                is_acceptable_in_the_filter = self.filter_SQP.add_to_filter((fy_next, v_next))

                if is_acceptable_in_the_filter:
                    print(f"Acceptable in the filter")
                    
                    v_curr = self.models.m_viol.feval(y_curr)
                    
                    mfy_curr = self.models.m_cf.model.model_polynomial.feval(y_curr)
                    mfy_next = self.models.m_cf.model.model_polynomial.feval(y_next)
                    fy_curr = self.models.m_cf.model.f[0]
                    
                    if mfy_curr - mfy_next >= self.constants['kappa_vartheta']*(v_curr**2):
                        print(f"With sufficient decrease")
                        rho = (fy_curr - fy_next)/(mfy_curr - mfy_next)
                        
                        if rho < self.constants['eta_1']:
                            print(f"Not good enough model")
                            radius = self.constants['gamma_1']*radius
                            Y = Y*1

                        else:
                            print(f"Good enough model")
                            Y = self.change_point(self.models, Y, y_next, radius)
                            
                            if rho >= self.constants['eta_2']:
                                radius = radius*self.constants['gamma_2']
                            else:
                                radius = radius*self.constants['gamma_1']
                                
                    else:
                        print(f"No sufficient decrease")
                        self.filter_SQP.add_to_filter((fy_next, v_next))
                        Y = self.change_point(self.models, Y, y_next, radius)
                            
                        if rho >= self.constants['eta_2']:
                            radius = radius*self.constants['gamma_2']
                        else:
                            radius = radius*self.constants['gamma_1']
                    
                    pass
                
                else:
                    print(f"Not acceptable in the filter")
                    radius = self.constants['gamma_0']*radius
                    Y = Y*1
            
            else:
                print(f"TRQP Incompatible")
                fy_curr = self.models.m_cf.model.f[0]
                v_curr = self.models.m_viol.feval(self.y_curr)
                _ = self.filter_SQP.add_to_filter((fy_curr, v_curr))
                
                Y = self.change_point(self.models, Y, y_next, radius)
                
        pass