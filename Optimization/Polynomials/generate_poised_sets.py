import numpy as np

class Ball: 
    def __init__(self, center:float=None, rad:float=None) -> None:
        self.center = center
        self.rad = rad
        
    def generate_vectors_with_uniform_angles(self, N:int) -> np.ndarray:
        
        # 1 center and N in a circle
        a = -1/(N-1)
        
        A = np.ones((N-1, N-1))*a
        np.fill_diagonal(A, 1)
        
        L = np.linalg.cholesky(A) 
        v = -np.sum(L.T, axis=1)[np.newaxis]
    
        vectors = np.concatenate((L.T, v.T), axis=1)

        return vectors
    
    
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