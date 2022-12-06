import numpy as np
import casadi as ca
from typing import List, Tuple, Optional

from .exceptions.TR_exceptions import IncorrectConstantsException
from .lagrange_polynomial import LagrangePolynomials
from .model_improvement import ModelImprovement
from .simulation_manager import EqualityConstraints, InequalityConstraints



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

            return constants

        def _check_constraints(eqcs:List[callable], ineqcs:List[callable]) -> Tuple:
            n_eqcs = len(eqcs)
            n_ineqcs = len(ineqcs)
            
            return n_eqcs, n_ineqcs

        self.constants = _check_constants(constants=constants)
        self.n_eqcs, self.n_ineqcs = _check_constraints(eqcs=eqcs, ineqcs=ineqcs)
        self.eqcs = EqualityConstraints(eqcs=eqcs); 
        self.ineqcs = InequalityConstraints(ineqcs=ineqcs)
        self.cf = cf
        self.dataset = dataset

        pass

    def __str__(self) -> str:
        return f"TrustRegionSQPFilter(n_eqcs={self.n_eqcs}, n_ineqcs={self.n_ineqcs})"

    def run(self):
        


        pass