import numpy as np
import casadi as ca
from typing import Tuple

class FilterSQP():
    def __init__(self, constants) -> None:
        self.filters = []
        
        self.constants = constants
    
    def add_to_filter(self, coordinate:Tuple[float, float]) -> bool:
        # coordinate is a combination of (OF, violation)
        
        self.filters.sort(key=lambda x: x[1])
        if len(self.filters) == 0:
            self.filters.append(coordinate)
            return True
        
        is_acceptable = False
            
        if coordinate[1] < (1 - self.constants['gamma_vartheta'])*self.filters[0][1] and coordinate[0] > self.filters[0][0] - self.constants['gamma_vartheta']*coordinate[1]:
            self.filters.append(coordinate)
            is_acceptable = True
        
        elif coordinate[1] > (1 - self.constants['gamma_vartheta'])*self.filters[-1][1] and coordinate[0] < self.filters[-1][0] - self.constants['gamma_vartheta']*coordinate[1]:
            self.filters.append(coordinate)
            is_acceptable = True
        
        else:
            ## Check main pareto front                
            curr_filter = []
            does_exist = False
            for i, coord in enumerate(self.filters):
                if coordinate[1] < (1 - self.constants['gamma_vartheta'])*coord[1] and coordinate[0] < coord[0] - self.constants['gamma_vartheta']*coordinate[1]:
                    curr_filter.append(coordinate)
                    is_acceptable = True
                    does_exist = True
                else:
                    if does_exist:
                        pass
                    else:
                        curr_filter.append(coord)
                        is_acceptable = True
                
            self.filters = curr_filter
        
        return is_acceptable
        