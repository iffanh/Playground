class SimulationManager():
    def __init__(self) -> None:
        pass

class CostFunction():
    def __init__(self) -> None:
        pass

class EqualityConstraints():
    def __init__(self, eqcs) -> None:
        self.eqcs = [EqualityConstraint(eq) for eq in eqcs]
        self.n_eqcs = len(self.eqcs)

class EqualityConstraint():
    def __init__(self, func:callable) -> None:
        self.func = func

    def __str__(self) -> str:
        return f"EqualityConstraint({self.func})"

class InequalityConstraints():
    def __init__(self, ineqcs) -> None:
        self.ineqcs = [InequalityConstraint(ineq) for ineq in ineqcs]
        self.n_ineqcs = len(self.ineqcs)

class InequalityConstraint():
    def __init__(self, func:callable) -> None:
        self.func = func

    def __str__(self) -> str:
        return f"InequalityConstraint({self.func})"