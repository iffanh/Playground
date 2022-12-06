import numpy as np
from typing import List

class CostFunctionModel():
    def __init__(self, Y:np.ndarray, fY:np.ndarray) -> None:
        self.Y = Y 
        self.fY = fY

    def __str__(self) -> str:
        return f"CostFunctionModel()"

class EqualityConstraintModels():
    def __init__(self, Ys:List[np.ndarray], fYs:List[np.ndarray]) -> None:
        self.Ys = Ys
        self.fYs = fYs

        self.meqcs = [EqualityConstraintModel(Y, fY) for Y, fY in zip(Ys, fYs)]
        self.n_meqcs = len(self.meqcs)

    def __str__(self) -> str:
        return f"EqualityConstraintModels(n = {self.n_meqcs})"

class EqualityConstraintModel():
    def __init__(self, Y:np.ndarray, fY:np.ndarray) -> None:
        self.Y = Y 
        self.fY = fY

    def __str__(self) -> str:
        return f"EqualityConstraintModel()"

class InequalityConstraintModels():
    def __init__(self, Ys:List[np.ndarray], fYs:List[np.ndarray]) -> None:
        self.Ys = Ys
        self.fYs = fYs

        self.mineqcs = [InequalityConstraintModel(Y, fY) for Y, fY in zip(Ys, fYs)]
        self.n_mineqcs = len(self.mineqcs)

    def __str__(self) -> str:
        return f"InequalityConstraintModels(n = {self.n_mineqcs})"

class InequalityConstraintModel():
    def __init__(self, Y:np.ndarray, fY:np.ndarray) -> None:
        self.Y = Y 
        self.fY = fY

    def __str__(self) -> str:
        return f"InequalityConstraintModel()"