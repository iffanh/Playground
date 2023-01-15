class SimulationManager():
    def __init__(self, cf, eqcs, ineqcs) -> None:
        
        self.eqcs = EqualityConstraints(eqcs=eqcs); 
        self.ineqcs = InequalityConstraints(ineqcs=ineqcs)
        self.cf = CostFunction(func=cf)
        
        pass
    
class CostFunction():
    def __init__(self, func:callable) -> None:
        self.number_of_function_calls = 0
        self._func = func
        pass
    
    def func(self, Y):
        self.number_of_function_calls += 1
        return self._func(Y)
    
    def __str__(self) -> str:
        return f"CostFunction(func = {self.func})"

class EqualityConstraints():
    def __init__(self, eqcs) -> None:
        self.eqcs = [EqualityConstraint(eq, ind) for ind, eq in enumerate(eqcs)]
        self.n_eqcs = len(self.eqcs)

    def __str__(self) -> str:
        return f"EqualityConstraints(n = {self.n_eqcs})"

class EqualityConstraint():
    def __init__(self, func:callable, index:int) -> None:
        self.func = func
        self.index = index

    def __str__(self) -> str:
        return f"EqualityConstraint(index={self.index}, {self.func})"

class InequalityConstraints():
    def __init__(self, ineqcs) -> None:
        self.ineqcs = [InequalityConstraint(ineq, ind) for ind, ineq in enumerate(ineqcs)]
        self.n_ineqcs = len(self.ineqcs)

    def __str__(self) -> str:
        return f"InequalityConstraints(n = {self.n_ineqcs})"

class InequalityConstraint():
    def __init__(self, func:callable, index:int) -> None:
        self.func = func
        self.index = index

    def __str__(self) -> str:
        return f"InequalityConstraint(index={self.index}, {self.func})"