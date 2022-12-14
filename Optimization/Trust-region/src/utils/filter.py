import numpy as np
import casadi as ca
from typing import Tuple

class FilterSQP():
    def __init__(self, constants) -> None:
        self.filters = []
        
        self.constants = constants
    
    def add_to_filter(self, coordinate:Tuple[float, float]):
        
        for i, coord in enumerate(self.filters):
                 
            if coordinate[1] < (1 - self.constants['gamma_vartheta'])*coord[1] or coordinate[0] < coord[0] - self.constants['gamma_vartheta']*coordinate[1]: 
                self.filters.append(coordinate)
                return True
            
            if coordinate[1] < (1 - self.constants['gamma_vartheta'])*coord[1] and coordinate[0] < coord[0] - self.constants['gamma_vartheta']*coordinate[1]: 
                self.filters[i] = coordinate
                return True
            
            if coordinate[1] >= (1 - self.constants['gamma_vartheta'])*coord[1] and coordinate[0] >= coord[0] - self.constants['gamma_vartheta']*coordinate[1]:
                return False
            
        self.filters.append(coordinate)
        self.filters.sort(key=lambda x: x[1])
        
        return True
        