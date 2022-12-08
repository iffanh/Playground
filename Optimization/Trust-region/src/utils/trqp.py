import numpy as np
import casadi as ca
from typing import Tuple

from .model_manager import ModelManager
from .TR_exceptions import TRQPIncompatible

class TRQP():
    def __init__(self, models:ModelManager, radius:float) -> None:
        self.sol, self.is_compatible = self.invoke_composite_step(models, radius)

    def invoke_composite_step(self, models:ModelManager, radius:float) -> Tuple[dict, bool]:
        ## construct TQRP problem (page 722, Chapter 15: Sequential Quadratic Programming)
        data = models.m_cf.model.y
        center = data[:,0]
        
        input_symbols = models.input_symbols

        # Cost function
        cf = models.m_cf.model.model_polynomial.symbol

        ubg = []
        lbg = []
        # Equality constraints
        eqcs = ca.vertcat(*[m.model.model_polynomial.symbol for m in models.m_eqcs.models])
        jc_eqcs = ca.jacobian(eqcs, input_symbols)
        eqcs_c = ca.Function('c_E', [input_symbols], [eqcs]) # equality constraint at center
        jc_eqcs_c = ca.Function('A_E', [input_symbols], [jc_eqcs]) # jacobian of equality constraint at center
        
        g_eq = ca.simplify(eqcs_c(center) + ca.mtimes(jc_eqcs_c(center), input_symbols - center))
        for _ in range(len(models.m_eqcs.models)):
            ubg.append(0.)
            lbg.append(0.)

        # Inequality constraints
        ineqcs = ca.vertcat(*[m.model.model_polynomial.symbol for m in models.m_ineqcs.models])
        jc_ineqcs = ca.jacobian(ineqcs, input_symbols)
        ineqcs_c = ca.Function('c_I', [input_symbols], [ineqcs]) # inequality constraint at center
        jc_ineqcs_c = ca.Function('A_I', [input_symbols], [jc_ineqcs]) # jacobian of inequality constraint at center
        
        g_ineq = ca.simplify(ineqcs_c(center) + ca.mtimes(jc_ineqcs_c(center), input_symbols - center))
        for _ in range(len(models.m_ineqcs.models)):
            ubg.append(ca.inf)
            lbg.append(0.)

        # tr radius constraints
        g_r = ca.norm_2(input_symbols - center)
        ubg.append(radius)
        lbg.append(0.)

        # construct NLP problem
        nlp = {
            'x': input_symbols,
            'f': cf, 
            'g': ca.vertcat(g_eq, g_ineq, g_r)
        }

        # solve TRQP problem
        solver = ca.nlpsol('TRQP', 'ipopt', nlp)

        sol = solver(x0=center, ubg=ubg, lbg=lbg)

        is_compatible = True
        try:
            if not solver.stats()['success']:
                print(f"fail with center as initial point")
                sol = solver(x0=center+radius/10, ubg=ubg, lbg=lbg)
                if not solver.stats()['success']:
                    TRQPIncompatible(f"TRQP is incompatible. Invoke restoration step")
        except TRQPIncompatible:
            sol = self.invoke_restoration_step(models, radius)
            is_compatible = False

        return sol, is_compatible

    def invoke_restoration_step(self, models:ModelManager, radius:float):
        return