import numpy as np
from typing import List
from .lagrange_polynomial import LagrangePolynomials

class CostFunctionModel():
    def __init__(self, input_symbols, Y:np.ndarray, fY:np.ndarray) -> None:
        self.model = LagrangePolynomials(input_symbols=input_symbols, pdegree=2)
        self.model.initialize(v=Y, f=fY, tr_radius=None)

    def __str__(self) -> str:
        return f"CostFunctionModel(model = {self.model.model_polynomial.symbol})"

class EqualityConstraintModels():
    def __init__(self, input_symbols, Y:np.ndarray, fYs:List[np.ndarray]) -> None:

        self.models = [EqualityConstraintModel(input_symbols, Y, fY, i) for i, fY in enumerate(fYs)]
        self.n = len(self.models)

    def __str__(self) -> str:
        return f"EqualityConstraintModels(n = {self.n})"

class EqualityConstraintModel():
    def __init__(self, input_symbols, Y:np.ndarray, fY:np.ndarray, index:int) -> None:
        self.model = LagrangePolynomials(input_symbols=input_symbols, pdegree=2)
        self.model.initialize(v=Y, f=fY, tr_radius=None)
        self.index = index

    def __str__(self) -> str:
        return f"EqualityConstraintModel(index={self.index}, model = {self.model.model_polynomial.symbol})"

class InequalityConstraintModels():
    def __init__(self, input_symbols, Y:List[np.ndarray], fYs:List[np.ndarray]) -> None:

        self.models = [InequalityConstraintModel(input_symbols, Y, fY, i) for i, fY in enumerate(fYs)]
        self.n = len(self.models)

    def __str__(self) -> str:
        return f"InequalityConstraintModels(n = {self.n})"

class InequalityConstraintModel():
    def __init__(self, input_symbols, Y:np.ndarray, fY:np.ndarray, index:int) -> None:
        self.model = LagrangePolynomials(input_symbols=input_symbols, pdegree=2)
        self.model.initialize(v=Y, f=fY, tr_radius=None)
        self.index = index

    def __str__(self) -> str:
        return f"InequalityConstraintModel(index={self.index}, model = {self.model.model_polynomial.symbol})"


class ModelManager():
    """
    Responsible for managing ALL the polynomial models
    """

    def __init__(self, input_symbols, m_cf:CostFunctionModel, m_eqcs:EqualityConstraintModels, m_ineqcs:InequalityConstraintModels) -> None:
        self.m_cf = m_cf
        self.m_eqcs = m_eqcs
        self.m_ineqcs = m_ineqcs
        self.input_symbols = input_symbols

    def __str__(self) -> str:
        return f"ModelManager(input_symbols = {self.input_symbols}, n_eqcs = {self.m_eqcs.n}, n_ineqcs = {self.m_ineqcs.n})"