import numpy as np
import casadi as ca
from typing import Tuple

class FilterSQP():
    def __init__(self, constants) -> None:
        self.filters = []
        
        self.constants = constants
    
    def add_to_filter(self, coordinate:Tuple[float, float]):
        
        print(coordinate)
        if len(self.filters) == 0:
            self.filters.append(coordinate)
            
            return True
        
        else:
            for i, coord in enumerate(self.filters):
                
                print(f"Condition 1: {coordinate[1] < (1 - self.constants['gamma_vartheta'])*coord[1]}")
                print(f"Condition 2: {coordinate[0] < coord[0] - self.constants['gamma_vartheta']*coordinate[1]}")
                
                if coordinate[1] < (1 - self.constants['gamma_vartheta'])*coord[1] or coordinate[0] < coord[0] - self.constants['gamma_vartheta']*coordinate[1]: 
                    self.filters.append(coordinate)
                    return True
                
            return False
        