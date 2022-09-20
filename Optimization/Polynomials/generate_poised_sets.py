"""
This script should be able to: 
- Find the "best" center data point.
- Find a \Lambda-poised set given initial interpolation set
- Calculate \Lambda
"""


import numpy as np

class Ball: 
    def __init__(self, center:float, rad:float) -> None:
        self.center = center
        self.rad = rad
        

class SampleSets:
    def __init__(self, y:np.ndarray) -> None:
        
        self.sorted_index = self._find_sorted_index_closest_point_to_center(y)
        self.y = y[:, self.sorted_index]
        self.ball = self._find_ball(self.y)
        
        
    def _find_ball(self, _y):
        
        center = _y[:,0]
        rad = np.linalg.norm(_y[:,0] - _y[:, -1])
        return Ball(center, rad)
    
    def _find_sorted_index_closest_point_to_center(self, y:np.ndarray) -> list:
        
        yave = self._average_point(y)
        
        dyn_list = []
        for i in range(y.shape[1]):
            dy = y[:, i] - yave
            dyn = np.linalg.norm(dy)
            dyn_list.append(dyn)
        
        sorted_index = sorted(range(len(dyn_list)), key=lambda k: dyn_list[k])
        
        return sorted_index
    
    def _average_point(self, y:np.ndarray) -> np.ndarray:
        return np.mean(y, axis=1)